"""
Cognitive State Space (CSS) Module for MARCO4

The CSS is the central repository for managing search branches in the
hierarchical architecture. It maintains a tree of partial grid hypotheses
and supports A* search with mass-based prioritization.
"""

import uuid
import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet
from dataclasses import dataclass, field
from enum import Enum

from .utils import Grid, grid_to_hash, copy_grid, count_filled_cells
from .dempster_shafer import MassFunction


class BranchStatus(Enum):
    """Status of a search branch."""
    ACTIVE = "active"
    PRUNED = "pruned"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class BranchNode:
    """
    A node in the Cognitive State Space representing a search branch.

    Each branch contains a (possibly partial) grid and associated
    mass/confidence information from experts.
    """
    id: str
    grid: Grid
    size: Tuple[int, int]  # (height, width)
    combined_mass: float
    expert_masses: Dict[str, MassFunction]  # expert_id -> mass function
    cell_masses: Dict[Tuple[int, int], MassFunction]  # (i,j) -> mass function
    cell_confidences: Dict[Tuple[int, int], float]  # (i,j) -> belief
    parent_id: Optional[str]
    children_ids: List[str]
    status: BranchStatus
    g: float  # Cost so far: -log(combined_mass)
    h: float  # Heuristic: estimated cost to solution
    f: float  # Total: g + h
    iteration_created: int
    no_progress_count: int = 0  # Iterations without new cells fixed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize computed fields."""
        if self.size is None:
            self.size = self.grid.shape
        if self.f is None or self.f == 0:
            self.f = self.g + self.h

    def filled_ratio(self) -> float:
        """Return fraction of cells filled."""
        total = self.grid.size
        filled = count_filled_cells(self.grid)
        return filled / total if total > 0 else 0.0

    def is_complete(self) -> bool:
        """Check if grid is fully filled."""
        return not np.any(self.grid == -1)

    def __lt__(self, other: 'BranchNode') -> bool:
        """For heap comparison, lower f is better."""
        return self.f < other.f


class CognitiveStateSpace:
    """
    Cognitive State Space for managing search branches.

    The CSS maintains a tree of partial grid hypotheses and provides
    A*-style search with mass-based prioritization.
    """

    def __init__(self):
        """Initialize empty CSS."""
        self.branches: Dict[str, BranchNode] = {}
        self.active_branches: Set[str] = set()
        self.pruned_branches: Set[str] = set()
        self.complete_branches: Set[str] = set()

        # Priority queue: (f_score, branch_id)
        # Using negative mass as priority (higher mass = lower priority value)
        self._priority_queue: List[Tuple[float, str]] = []

        # Index for quick lookup
        self._grid_hash_to_id: Dict[str, str] = {}

        # Statistics
        self.total_branches_created = 0
        self.total_branches_pruned = 0

    def _generate_id(self) -> str:
        """Generate unique branch ID."""
        return str(uuid.uuid4())[:12]

    def add_branch(
        self,
        grid: Grid,
        combined_mass: float,
        expert_masses: Optional[Dict[str, MassFunction]] = None,
        cell_masses: Optional[Dict[Tuple[int, int], MassFunction]] = None,
        cell_confidences: Optional[Dict[Tuple[int, int], float]] = None,
        parent_id: Optional[str] = None,
        heuristic: float = 0.0,
        iteration: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new branch to the CSS.

        Args:
            grid: The grid (partial or complete)
            combined_mass: Overall mass for this branch
            expert_masses: Mass functions from each expert
            cell_masses: Mass functions for each cell
            cell_confidences: Belief values for each cell
            parent_id: ID of parent branch (if any)
            heuristic: Heuristic estimate to solution
            iteration: Current iteration number
            metadata: Additional branch metadata

        Returns:
            ID of the new branch
        """
        # Check for duplicate grids
        grid_hash = grid_to_hash(grid)
        if grid_hash in self._grid_hash_to_id:
            existing_id = self._grid_hash_to_id[grid_hash]
            existing = self.branches.get(existing_id)
            if existing and existing.combined_mass < combined_mass:
                # Update existing with better mass
                existing.combined_mass = combined_mass
                existing.g = -np.log(max(combined_mass, 1e-10))
                existing.f = existing.g + existing.h
                self._update_priority(existing_id)
            return existing_id

        branch_id = self._generate_id()

        # Compute g (cost so far) from mass
        g = -np.log(max(combined_mass, 1e-10))

        # Determine status
        is_complete = not np.any(grid == -1)
        status = BranchStatus.COMPLETE if is_complete else BranchStatus.ACTIVE

        node = BranchNode(
            id=branch_id,
            grid=copy_grid(grid),
            size=grid.shape,
            combined_mass=combined_mass,
            expert_masses=expert_masses or {},
            cell_masses=cell_masses or {},
            cell_confidences=cell_confidences or {},
            parent_id=parent_id,
            children_ids=[],
            status=status,
            g=g,
            h=heuristic,
            f=g + heuristic,
            iteration_created=iteration,
            metadata=metadata or {}
        )

        # Store in branches
        self.branches[branch_id] = node
        self._grid_hash_to_id[grid_hash] = branch_id
        self.total_branches_created += 1

        # Update parent's children list
        if parent_id and parent_id in self.branches:
            self.branches[parent_id].children_ids.append(branch_id)

        # Add to appropriate set
        if status == BranchStatus.COMPLETE:
            self.complete_branches.add(branch_id)
        else:
            self.active_branches.add(branch_id)
            heapq.heappush(self._priority_queue, (node.f, branch_id))

        return branch_id

    def _update_priority(self, branch_id: str) -> None:
        """Re-add branch to priority queue with updated priority."""
        branch = self.branches.get(branch_id)
        if branch and branch.status == BranchStatus.ACTIVE:
            heapq.heappush(self._priority_queue, (branch.f, branch_id))

    def get_branch(self, branch_id: str) -> Optional[BranchNode]:
        """Get a branch by ID."""
        return self.branches.get(branch_id)

    def get_highest_mass_branch(
        self,
        status: Optional[BranchStatus] = BranchStatus.ACTIVE
    ) -> Optional[BranchNode]:
        """
        Return branch with highest combined_mass (lowest f score).

        Args:
            status: Filter by status (None for any)

        Returns:
            Branch with highest mass, or None if no matching branches
        """
        while self._priority_queue:
            _, branch_id = heapq.heappop(self._priority_queue)
            branch = self.branches.get(branch_id)

            if branch is None:
                continue

            if status is not None and branch.status != status:
                continue

            # Re-add to queue for future access
            heapq.heappush(self._priority_queue, (branch.f, branch_id))
            return branch

        return None

    def get_top_branches(
        self,
        n: int,
        status: Optional[BranchStatus] = BranchStatus.ACTIVE
    ) -> List[BranchNode]:
        """
        Get top N branches by mass.

        Args:
            n: Number of branches to return
            status: Filter by status

        Returns:
            List of branches sorted by mass (highest first)
        """
        if status == BranchStatus.ACTIVE:
            branch_ids = self.active_branches
        elif status == BranchStatus.COMPLETE:
            branch_ids = self.complete_branches
        elif status == BranchStatus.PRUNED:
            branch_ids = self.pruned_branches
        else:
            branch_ids = set(self.branches.keys())

        branches = [self.branches[bid] for bid in branch_ids
                    if bid in self.branches]
        branches.sort(key=lambda b: b.combined_mass, reverse=True)

        return branches[:n]

    def prune_branch(self, branch_id: str, recursive: bool = True) -> int:
        """
        Mark a branch as pruned.

        Args:
            branch_id: ID of branch to prune
            recursive: If True, also prune all descendants

        Returns:
            Number of branches pruned
        """
        count = 0
        branch = self.branches.get(branch_id)

        if branch is None or branch.status == BranchStatus.PRUNED:
            return count

        branch.status = BranchStatus.PRUNED
        self.active_branches.discard(branch_id)
        self.complete_branches.discard(branch_id)
        self.pruned_branches.add(branch_id)
        count += 1
        self.total_branches_pruned += 1

        if recursive:
            for child_id in branch.children_ids:
                count += self.prune_branch(child_id, recursive=True)

        return count

    def mark_complete(self, branch_id: str) -> bool:
        """
        Mark a branch as complete (solution found).

        Args:
            branch_id: ID of branch to mark

        Returns:
            True if successfully marked
        """
        branch = self.branches.get(branch_id)
        if branch is None:
            return False

        branch.status = BranchStatus.COMPLETE
        self.active_branches.discard(branch_id)
        self.complete_branches.add(branch_id)
        return True

    def update_branch(
        self,
        branch_id: str,
        grid: Optional[Grid] = None,
        combined_mass: Optional[float] = None,
        cell_masses: Optional[Dict[Tuple[int, int], MassFunction]] = None,
        cell_confidences: Optional[Dict[Tuple[int, int], float]] = None,
        heuristic: Optional[float] = None,
        no_progress_increment: bool = False
    ) -> bool:
        """
        Update an existing branch.

        Args:
            branch_id: ID of branch to update
            grid: New grid (optional)
            combined_mass: New mass (optional)
            cell_masses: New cell masses (optional)
            cell_confidences: New cell confidences (optional)
            heuristic: New heuristic (optional)
            no_progress_increment: If True, increment no_progress_count

        Returns:
            True if successfully updated
        """
        branch = self.branches.get(branch_id)
        if branch is None:
            return False

        if grid is not None:
            # Update grid hash index
            old_hash = grid_to_hash(branch.grid)
            new_hash = grid_to_hash(grid)
            if old_hash in self._grid_hash_to_id:
                del self._grid_hash_to_id[old_hash]
            self._grid_hash_to_id[new_hash] = branch_id

            branch.grid = copy_grid(grid)
            branch.size = grid.shape

        if combined_mass is not None:
            branch.combined_mass = combined_mass
            branch.g = -np.log(max(combined_mass, 1e-10))

        if cell_masses is not None:
            branch.cell_masses.update(cell_masses)

        if cell_confidences is not None:
            branch.cell_confidences.update(cell_confidences)

        if heuristic is not None:
            branch.h = heuristic

        if no_progress_increment:
            branch.no_progress_count += 1

        # Recompute f
        branch.f = branch.g + branch.h

        # Update priority queue
        if branch.status == BranchStatus.ACTIVE:
            self._update_priority(branch_id)

        return True

    def get_best_complete_branch(self) -> Optional[BranchNode]:
        """
        Get the complete branch with highest mass.

        Returns:
            Best complete branch, or None if no complete branches
        """
        if not self.complete_branches:
            return None

        best = None
        best_mass = -1.0

        for branch_id in self.complete_branches:
            branch = self.branches.get(branch_id)
            if branch and branch.combined_mass > best_mass:
                best = branch
                best_mass = branch.combined_mass

        return best

    def get_branch_path(self, branch_id: str) -> List[BranchNode]:
        """
        Get the path from root to this branch.

        Args:
            branch_id: Target branch ID

        Returns:
            List of branches from root to target
        """
        path = []
        current_id = branch_id

        while current_id is not None:
            branch = self.branches.get(current_id)
            if branch is None:
                break
            path.append(branch)
            current_id = branch.parent_id

        path.reverse()
        return path

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get CSS statistics.

        Returns:
            Dictionary of statistics
        """
        active_masses = [
            self.branches[bid].combined_mass
            for bid in self.active_branches
            if bid in self.branches
        ]

        complete_masses = [
            self.branches[bid].combined_mass
            for bid in self.complete_branches
            if bid in self.branches
        ]

        return {
            'total_branches': len(self.branches),
            'active_branches': len(self.active_branches),
            'complete_branches': len(self.complete_branches),
            'pruned_branches': len(self.pruned_branches),
            'total_created': self.total_branches_created,
            'total_pruned': self.total_branches_pruned,
            'avg_active_mass': np.mean(active_masses) if active_masses else 0.0,
            'max_active_mass': max(active_masses) if active_masses else 0.0,
            'avg_complete_mass': np.mean(complete_masses) if complete_masses else 0.0,
            'max_complete_mass': max(complete_masses) if complete_masses else 0.0,
        }

    def clear_pruned(self) -> int:
        """
        Remove pruned branches from memory.

        Returns:
            Number of branches removed
        """
        count = len(self.pruned_branches)

        for branch_id in list(self.pruned_branches):
            branch = self.branches.get(branch_id)
            if branch:
                # Remove from grid hash index
                grid_hash = grid_to_hash(branch.grid)
                if grid_hash in self._grid_hash_to_id:
                    del self._grid_hash_to_id[grid_hash]

                # Remove from parent's children
                if branch.parent_id and branch.parent_id in self.branches:
                    parent = self.branches[branch.parent_id]
                    if branch_id in parent.children_ids:
                        parent.children_ids.remove(branch_id)

            del self.branches[branch_id]

        self.pruned_branches.clear()
        return count

    def __len__(self) -> int:
        """Return total number of branches."""
        return len(self.branches)

    def __contains__(self, branch_id: str) -> bool:
        """Check if branch exists."""
        return branch_id in self.branches
