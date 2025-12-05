"""
Meta Control Unit (MCU) for MARCO4

The MCU orchestrates the two-level hierarchical architecture:
1. Collects per-cell logits from experts
2. Combines logits using Dempster-Shafer theory
3. Maintains multiple candidate values per cell until pruned
4. Creates multiple branches when cells have competing candidates
5. Manages the A* search through the Cognitive State Space

Search Strategy:
- When a cell has multiple viable candidates (belief > branch_threshold),
  the MCU creates separate branches for each candidate
- Each branch fixes a different value and continues exploration
- Branches are prioritized by their combined mass (A* search)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .dempster_shafer import (
    MassFunction, THETA, dempster_combine, dempster_combine_multiple,
    compute_belief, compute_plausibility, get_best_color,
    get_conflict_level, get_pignistic_distribution
)
from .css import CognitiveStateSpace, BranchNode, BranchStatus
from .expert import Expert, ExpertOutput, CellLogits
from .utils import (
    Grid, copy_grid, create_empty_grid, is_complete_grid,
    count_filled_cells, count_empty_cells
)
from .config import MARCO4Config, DEFAULT_CONFIG


logger = logging.getLogger(__name__)


@dataclass
class CellState:
    """
    State of a single cell, maintaining multiple candidate values.

    Before pruning, a cell can have multiple possible values with
    associated beliefs. Once a value exceeds the confidence threshold,
    the cell is "fixed" to that value.
    """
    mass_function: MassFunction  # D-S mass function over colors
    fixed_value: Optional[int] = None  # Set when confidence exceeds threshold
    belief: float = 0.0  # Belief in the best color

    def get_best_color(self) -> Tuple[Optional[int], float]:
        """Get the most likely color and its belief."""
        return get_best_color(self.mass_function)

    def is_fixed(self) -> bool:
        """Check if cell has been fixed to a value."""
        return self.fixed_value is not None


@dataclass
class GridState:
    """
    State of the entire grid with per-cell mass functions.

    Maintains uncertainty at each cell until pruned/fixed.
    """
    cell_states: Dict[Tuple[int, int], CellState]
    shape: Tuple[int, int]
    overall_mass: float = 1.0
    avg_conflict: float = 0.0

    def to_grid(self) -> Grid:
        """Convert to numpy grid, using best color for each cell."""
        h, w = self.shape
        grid = create_empty_grid(h, w)
        for (i, j), cell_state in self.cell_states.items():
            if cell_state.is_fixed():
                grid[i, j] = cell_state.fixed_value
            else:
                best_color, _ = cell_state.get_best_color()
                grid[i, j] = best_color if best_color is not None else -1
        return grid

    def to_partial_grid(self, confidence_threshold: float) -> Grid:
        """Convert to partial grid, only including high-confidence cells."""
        h, w = self.shape
        grid = create_empty_grid(h, w)
        for (i, j), cell_state in self.cell_states.items():
            if cell_state.is_fixed():
                grid[i, j] = cell_state.fixed_value
            else:
                best_color, belief = cell_state.get_best_color()
                if belief >= confidence_threshold and best_color is not None:
                    grid[i, j] = best_color
        return grid

    def get_fixed_count(self) -> int:
        """Count number of fixed cells."""
        return sum(1 for cs in self.cell_states.values() if cs.is_fixed())

    def is_complete(self) -> bool:
        """Check if all cells are fixed."""
        return self.get_fixed_count() == len(self.cell_states)

    def get_uncertain_cells(self, branch_threshold: float) -> List[Tuple[Tuple[int, int], List[Tuple[int, float]]]]:
        """
        Find cells with multiple viable candidates above threshold.

        Returns:
            List of ((row, col), [(color, belief), ...]) for uncertain cells
        """
        uncertain = []
        for (i, j), cell_state in self.cell_states.items():
            if cell_state.is_fixed():
                continue

            # Get all candidates above threshold
            candidates = []
            pignistic = get_pignistic_distribution(cell_state.mass_function)
            for color, prob in pignistic.items():
                belief = compute_belief(color, cell_state.mass_function)
                if belief >= branch_threshold:
                    candidates.append((color, belief))

            # Sort by belief descending
            candidates.sort(key=lambda x: x[1], reverse=True)

            if len(candidates) > 1:
                uncertain.append(((i, j), candidates))

        return uncertain

    def get_most_uncertain_cell(self, branch_threshold: float) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, float]]]]:
        """
        Find the cell with highest uncertainty (most competing candidates).

        Returns:
            ((row, col), [(color, belief), ...]) or None if no uncertain cells
        """
        uncertain = self.get_uncertain_cells(branch_threshold)
        if not uncertain:
            return None

        # Find cell with most candidates or highest entropy
        # Using number of candidates as proxy for uncertainty
        return max(uncertain, key=lambda x: len(x[1]))


@dataclass
class CombinedResult:
    """Result from D-S combination of expert outputs."""
    grid_state: GridState
    conflict_level: float


@dataclass
class SolveResult:
    """Result from solving an ARC task."""
    solution: Optional[Grid]
    confidence: float
    iterations: int
    branches_explored: int
    status: str  # 'solved', 'max_iterations', 'no_solution'


class MCU:
    """
    Meta Control Unit for MARCO4.

    Orchestrates per-cell D-S combination with multiple candidate values.
    """

    def __init__(
        self,
        experts: List[Expert],
        config: Optional[MARCO4Config] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ):
        self.experts = experts
        self.config = config or DEFAULT_CONFIG
        self.css = CognitiveStateSpace()
        self.parallel = parallel
        self.max_workers = max_workers or len(experts)
        self.progress_callback = progress_callback

    def solve(
        self,
        problem: Any,
        target_size: Optional[Tuple[int, int]] = None,
        max_iterations: Optional[int] = None
    ) -> SolveResult:
        """
        Main solving loop using A* search with per-cell D-S combination.

        Args:
            problem: ARC task data
            target_size: Expected output size (height, width)
            max_iterations: Max iterations (default from config)

        Returns:
            SolveResult with solution grid and metadata
        """
        if max_iterations is None:
            max_iterations = self.config.search.max_iterations

        if target_size is None:
            target_size = self._infer_target_size(problem)

        # Initialize with empty grid
        initial_grid = create_empty_grid(*target_size)
        root_id = self.css.add_branch(
            grid=initial_grid,
            combined_mass=1.0,
            iteration=0
        )

        iteration = 0
        best_solution: Optional[Grid] = None
        best_confidence = 0.0

        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Iteration {iteration}/{max_iterations}")

            # Get highest-mass active branch
            branch = self.css.get_highest_mass_branch(status=BranchStatus.ACTIVE)
            if branch is None:
                logger.info("No more active branches")
                break

            # Check if already complete
            if branch.is_complete():
                if branch.combined_mass > best_confidence:
                    best_solution = branch.grid
                    best_confidence = branch.combined_mass

                if best_confidence >= self.config.confidence.solution_threshold:
                    return SolveResult(
                        solution=best_solution,
                        confidence=best_confidence,
                        iterations=iteration,
                        branches_explored=self.css.total_branches_created,
                        status='solved'
                    )

                self.css.mark_complete(branch.id)
                continue

            # Collect per-cell logits from all experts
            expert_outputs = self._collect_expert_outputs(problem, branch.grid)

            if not expert_outputs:
                self.css.prune_branch(branch.id)
                continue

            # Combine using D-S at cell level
            combined = self._combine_expert_outputs(expert_outputs, target_size)

            # Check for high conflict -> prune
            if combined.conflict_level > self.config.pruning.max_conflict:
                logger.debug(f"Pruning branch {branch.id}: high conflict {combined.conflict_level:.3f}")
                self.css.prune_branch(branch.id)
                continue

            # Fix high-confidence cells
            grid_state = self._fix_confident_cells(
                combined.grid_state,
                self.config.confidence.high_confidence
            )

            # Extract partial grid for progress tracking
            partial_grid = grid_state.to_partial_grid(
                self.config.confidence.high_confidence
            )

            # Check if solution found
            if grid_state.is_complete():
                solution_grid = grid_state.to_grid()
                if grid_state.overall_mass > best_confidence:
                    best_solution = solution_grid
                    best_confidence = grid_state.overall_mass

                # Invoke progress callback before returning (so we capture final state)
                if self.progress_callback is not None:
                    self.progress_callback(
                        iteration=iteration,
                        grid_state=grid_state,
                        partial_grid=solution_grid,  # Use complete solution
                        best_solution=best_solution,
                        best_confidence=best_confidence,
                        conflict_level=combined.conflict_level,
                        active_branches=self.css.get_statistics()['active_branches'],
                        total_branches=self.css.total_branches_created
                    )

                if best_confidence >= self.config.confidence.solution_threshold:
                    return SolveResult(
                        solution=best_solution,
                        confidence=best_confidence,
                        iterations=iteration,
                        branches_explored=self.css.total_branches_created,
                        status='solved'
                    )

            # Invoke progress callback if provided (for incomplete grids)
            elif self.progress_callback is not None:
                self.progress_callback(
                    iteration=iteration,
                    grid_state=grid_state,
                    partial_grid=partial_grid,
                    best_solution=best_solution,
                    best_confidence=best_confidence,
                    conflict_level=combined.conflict_level,
                    active_branches=self.css.get_statistics()['active_branches'],
                    total_branches=self.css.total_branches_created
                )

            # Check progress
            old_filled = count_filled_cells(branch.grid)
            new_filled = count_filled_cells(partial_grid)

            if new_filled <= old_filled:
                self.css.update_branch(branch.id, no_progress_increment=True)
                if branch.no_progress_count >= self.config.pruning.no_progress_rounds:
                    self.css.prune_branch(branch.id)
                    continue

            # MCU-level pruning check
            if grid_state.overall_mass < self.config.pruning.mcu_prune_threshold:
                logger.debug(f"Pruning branch {branch.id}: low mass {grid_state.overall_mass:.4f}")
                self.css.prune_branch(branch.id)
                continue

            # MCU-driven branching: find cells with competing candidates
            uncertain_cell = grid_state.get_most_uncertain_cell(
                self.config.search.branch_threshold
            )

            if uncertain_cell is not None:
                # Create multiple branches for competing candidates
                (cell_row, cell_col), candidates = uncertain_cell

                # Limit number of branches
                top_candidates = candidates[:self.config.search.max_branches_per_cell]

                logger.debug(
                    f"Creating {len(top_candidates)} branches for cell ({cell_row}, {cell_col}): "
                    f"{[(c, f'{b:.3f}') for c, b in top_candidates]}"
                )

                for color, belief in top_candidates:
                    # Create a new partial grid with this cell fixed
                    branch_grid = copy_grid(partial_grid)
                    branch_grid[cell_row, cell_col] = color

                    # Compute branch-specific mass (weighted by candidate belief)
                    branch_mass = grid_state.overall_mass * belief

                    heuristic = self._compute_heuristic(branch_grid, grid_state)
                    self.css.add_branch(
                        grid=branch_grid,
                        combined_mass=branch_mass,
                        cell_masses={pos: cs.mass_function for pos, cs in grid_state.cell_states.items()},
                        cell_confidences={pos: cs.belief for pos, cs in grid_state.cell_states.items()},
                        parent_id=branch.id,
                        heuristic=heuristic,
                        iteration=iteration
                    )
            else:
                # No uncertain cells - add single branch with partial grid
                heuristic = self._compute_heuristic(partial_grid, grid_state)
                self.css.add_branch(
                    grid=partial_grid,
                    combined_mass=grid_state.overall_mass,
                    cell_masses={pos: cs.mass_function for pos, cs in grid_state.cell_states.items()},
                    cell_confidences={pos: cs.belief for pos, cs in grid_state.cell_states.items()},
                    parent_id=branch.id,
                    heuristic=heuristic,
                    iteration=iteration
                )

            # Mark original as expanded
            self.css.update_branch(branch.id, heuristic=float('inf'))

        # Return best found
        if best_solution is not None:
            return SolveResult(
                solution=best_solution,
                confidence=best_confidence,
                iterations=iteration,
                branches_explored=self.css.total_branches_created,
                status='max_iterations' if iteration >= max_iterations else 'solved'
            )

        # Check complete branches
        best_complete = self.css.get_best_complete_branch()
        if best_complete is not None:
            return SolveResult(
                solution=best_complete.grid,
                confidence=best_complete.combined_mass,
                iterations=iteration,
                branches_explored=self.css.total_branches_created,
                status='best_effort'
            )

        return SolveResult(
            solution=None,
            confidence=0.0,
            iterations=iteration,
            branches_explored=self.css.total_branches_created,
            status='no_solution'
        )

    def _collect_expert_outputs(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> List[ExpertOutput]:
        """Collect per-cell logits from all experts."""
        if self.parallel and len(self.experts) > 1:
            return self._collect_parallel(problem, partial_grid)
        else:
            return self._collect_sequential(problem, partial_grid)

    def _collect_sequential(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> List[ExpertOutput]:
        """Sequential expert collection."""
        outputs = []
        for expert in self.experts:
            try:
                output = expert.get_output(problem, partial_grid)
                outputs.append(output)
            except Exception as e:
                logger.error(f"Expert {expert.expert_id} failed: {e}")
        return outputs

    def _collect_parallel(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> List[ExpertOutput]:
        """Parallel expert collection."""
        outputs = []

        def run_expert(expert: Expert) -> Optional[ExpertOutput]:
            try:
                return expert.get_output(problem, partial_grid)
            except Exception as e:
                logger.error(f"Expert {expert.expert_id} failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_expert, exp): exp for exp in self.experts}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    outputs.append(result)

        return outputs

    def _combine_expert_outputs(
        self,
        expert_outputs: List[ExpertOutput],
        target_size: Tuple[int, int]
    ) -> CombinedResult:
        """
        Combine per-cell logits from multiple experts using D-S.

        For each cell:
        1. Convert each expert's logits to mass function
        2. Combine mass functions using Dempster's rule
        3. Track conflict level
        """
        h, w = target_size
        cell_states: Dict[Tuple[int, int], CellState] = {}
        total_conflict = 0.0
        n_cells = 0

        for i in range(h):
            for j in range(w):
                # Collect mass functions from all experts for this cell
                cell_masses: List[MassFunction] = []

                for output in expert_outputs:
                    mass_funcs = output.get_mass_functions()
                    if (i, j) in mass_funcs:
                        cell_masses.append(mass_funcs[(i, j)])

                if not cell_masses:
                    # No expert data for this cell
                    cell_states[(i, j)] = CellState(
                        mass_function={THETA: 1.0},
                        belief=0.0
                    )
                    continue

                # Combine using D-S
                if len(cell_masses) == 1:
                    combined = cell_masses[0]
                else:
                    combined = dempster_combine_multiple(cell_masses)

                    # Track conflict (between first two)
                    conflict = get_conflict_level(cell_masses[0], cell_masses[1])
                    total_conflict += conflict
                    n_cells += 1

                # Get best color and belief
                best_color, belief = get_best_color(combined)

                cell_states[(i, j)] = CellState(
                    mass_function=combined,
                    belief=belief
                )

        avg_conflict = total_conflict / n_cells if n_cells > 0 else 0.0

        # Compute overall mass as geometric mean of beliefs
        beliefs = [cs.belief for cs in cell_states.values() if cs.belief > 0]
        if beliefs:
            log_beliefs = [np.log(max(b, 1e-10)) for b in beliefs]
            overall_mass = np.exp(np.mean(log_beliefs))
        else:
            overall_mass = 0.0

        grid_state = GridState(
            cell_states=cell_states,
            shape=target_size,
            overall_mass=overall_mass,
            avg_conflict=avg_conflict
        )

        return CombinedResult(
            grid_state=grid_state,
            conflict_level=avg_conflict
        )

    def _fix_confident_cells(
        self,
        grid_state: GridState,
        confidence_threshold: float
    ) -> GridState:
        """
        Fix cells that exceed the confidence threshold.

        Once fixed, a cell's value won't change in future iterations.
        """
        for (i, j), cell_state in grid_state.cell_states.items():
            if cell_state.is_fixed():
                continue

            best_color, belief = cell_state.get_best_color()
            if belief >= confidence_threshold and best_color is not None:
                cell_state.fixed_value = best_color

        return grid_state

    def _compute_heuristic(
        self,
        partial_grid: Grid,
        grid_state: Optional[GridState] = None
    ) -> float:
        """
        Compute A* heuristic based on D-S mass uncertainty.

        The heuristic combines:
        1. Fraction of unfilled cells (basic progress measure)
        2. Average uncertainty of unfilled cells based on D-S masses

        For each unfilled cell, uncertainty is measured as:
        - 1 - max_pignistic_prob: high when no color dominates

        Returns value in [0, 1] where 0 = complete solution, 1 = maximum uncertainty.
        """
        h, w = partial_grid.shape
        total_cells = h * w

        if total_cells == 0:
            return 0.0

        # Count empty cells
        empty_cells = count_empty_cells(partial_grid)

        # If grid is complete, heuristic is 0
        if empty_cells == 0:
            return 0.0

        # Basic heuristic: fraction of empty cells
        empty_fraction = empty_cells / total_cells

        # If no grid_state provided, use simple heuristic
        if grid_state is None:
            return empty_fraction

        # Compute uncertainty-based heuristic using D-S masses
        total_uncertainty = 0.0
        unfilled_count = 0

        for (i, j), cell_state in grid_state.cell_states.items():
            # Skip fixed cells
            if cell_state.is_fixed():
                continue

            # Skip cells already filled in partial grid
            if partial_grid[i, j] >= 0:
                continue

            unfilled_count += 1

            # Get pignistic probability distribution for colors 0-9
            pignistic = get_pignistic_distribution(cell_state.mass_function)

            if pignistic:
                # Max probability for any color
                max_prob = max(pignistic.values())
                # Uncertainty = 1 - max_prob (high when distribution is uniform)
                cell_uncertainty = 1.0 - max_prob
            else:
                # Complete uncertainty
                cell_uncertainty = 1.0

            total_uncertainty += cell_uncertainty

        # Average uncertainty per unfilled cell
        if unfilled_count > 0:
            avg_uncertainty = total_uncertainty / unfilled_count
        else:
            avg_uncertainty = 0.0

        # Combine empty fraction and uncertainty
        # Weight: 50% progress (empty cells), 50% uncertainty
        heuristic = 0.5 * empty_fraction + 0.5 * avg_uncertainty * empty_fraction

        return heuristic

    def _infer_target_size(self, problem: Any) -> Tuple[int, int]:
        """Infer target size from training examples."""
        if hasattr(problem, 'train') and problem.train:
            sizes = []
            for example in problem.train:
                output = example.get('output', [])
                if output:
                    h = len(output)
                    w = len(output[0]) if output else 0
                    sizes.append((h, w))

            if sizes:
                from collections import Counter
                return Counter(sizes).most_common(1)[0][0]

        return (3, 3)

    def get_statistics(self) -> Dict[str, Any]:
        """Get MCU and CSS statistics."""
        css_stats = self.css.get_statistics()
        return {
            'css': css_stats,
            'num_experts': len(self.experts),
            'config': self.config.to_dict()
        }
