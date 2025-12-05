"""
Size Resolution Module for MARCO4

Implements hierarchical size disagreement resolution when experts
propose grids of different dimensions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from .utils import Grid, copy_grid, pad_grid
from .dempster_shafer import MassFunction, dempster_combine_multiple
from .config import MARCO4Config


@dataclass
class SizeCandidate:
    """A candidate output size with supporting evidence."""
    size: Tuple[int, int]  # (height, width)
    expert_count: int  # Number of experts supporting this size
    total_mass: float  # Sum of masses from supporting experts
    avg_confidence: float  # Average confidence
    grids: List[Tuple[Grid, float, Dict]]  # (grid, mass, cell_masses)
    expert_ids: List[str]  # IDs of supporting experts


@dataclass
class SizeResolutionResult:
    """Result of size resolution process."""
    selected_size: Tuple[int, int]
    confidence: float
    method: str  # 'voting', 'content', 'joint', 'requery'
    candidates: List[SizeCandidate]
    combined_grids: List[Tuple[Grid, float, Dict]]  # Grids at selected size


class SizeResolver:
    """
    Resolves size disagreements between expert outputs.

    Uses a three-level hierarchical approach:
    1. Size voting based on expert count and mass
    2. Content-based scoring within each size
    3. Joint scoring across sizes

    Falls back to re-querying experts with size constraint if needed.
    """

    def __init__(self, config: Optional[MARCO4Config] = None):
        """
        Initialize size resolver.

        Args:
            config: MARCO4 configuration (uses defaults if None)
        """
        self.config = config or MARCO4Config()

    def resolve(
        self,
        expert_results: Dict[str, List[Tuple[Grid, float, Dict]]],
        problem: Optional[Any] = None
    ) -> SizeResolutionResult:
        """
        Resolve size disagreements among expert results.

        Args:
            expert_results: Dict mapping expert_id to list of (grid, mass, cell_masses)
            problem: Original ARC problem (for re-query if needed)

        Returns:
            SizeResolutionResult with selected size and combined grids
        """
        # Group grids by size
        candidates = self._group_by_size(expert_results)

        if len(candidates) == 0:
            # No valid grids
            return SizeResolutionResult(
                selected_size=(0, 0),
                confidence=0.0,
                method='none',
                candidates=[],
                combined_grids=[]
            )

        if len(candidates) == 1:
            # All experts agree on size
            size, candidate = list(candidates.items())[0]
            return SizeResolutionResult(
                selected_size=size,
                confidence=1.0,
                method='unanimous',
                candidates=[candidate],
                combined_grids=candidate.grids
            )

        # Level 1: Size voting
        voted_size, vote_confidence = self._level1_voting(candidates)

        if vote_confidence >= self.config.size_resolution.size_vote_threshold:
            return SizeResolutionResult(
                selected_size=voted_size,
                confidence=vote_confidence,
                method='voting',
                candidates=list(candidates.values()),
                combined_grids=candidates[voted_size].grids
            )

        # Level 2: Content-based scoring per size
        scored_candidates = self._level2_content_scoring(candidates)

        # Level 3: Joint scoring
        best_size, best_score = self._level3_joint_scoring(scored_candidates)

        if best_score > 0:
            return SizeResolutionResult(
                selected_size=best_size,
                confidence=best_score,
                method='joint',
                candidates=list(candidates.values()),
                combined_grids=candidates[best_size].grids
            )

        # Fallback: use most common size
        fallback_size = max(candidates.keys(),
                            key=lambda s: candidates[s].expert_count)

        return SizeResolutionResult(
            selected_size=fallback_size,
            confidence=0.5,
            method='fallback',
            candidates=list(candidates.values()),
            combined_grids=candidates[fallback_size].grids
        )

    def _group_by_size(
        self,
        expert_results: Dict[str, List[Tuple[Grid, float, Dict]]]
    ) -> Dict[Tuple[int, int], SizeCandidate]:
        """
        Group expert grids by their size.

        Args:
            expert_results: Expert outputs

        Returns:
            Dict mapping size to SizeCandidate
        """
        size_groups: Dict[Tuple[int, int], List[Tuple[Grid, float, Dict, str]]] = \
            defaultdict(list)

        for expert_id, grids in expert_results.items():
            for grid, mass, cell_masses in grids:
                size = grid.shape
                size_groups[size].append((grid, mass, cell_masses, expert_id))

        candidates = {}
        for size, items in size_groups.items():
            grids = [(g, m, c) for g, m, c, _ in items]
            expert_ids = list(set(e for _, _, _, e in items))
            masses = [m for _, m, _, _ in items]

            candidates[size] = SizeCandidate(
                size=size,
                expert_count=len(expert_ids),
                total_mass=sum(masses),
                avg_confidence=np.mean(masses) if masses else 0.0,
                grids=grids,
                expert_ids=expert_ids
            )

        return candidates

    def _level1_voting(
        self,
        candidates: Dict[Tuple[int, int], SizeCandidate]
    ) -> Tuple[Tuple[int, int], float]:
        """
        Level 1: Vote on size based on expert count and mass.

        Args:
            candidates: Size candidates

        Returns:
            (best_size, confidence)
        """
        total_experts = sum(c.expert_count for c in candidates.values())
        total_mass = sum(c.total_mass for c in candidates.values())

        if total_experts == 0 or total_mass == 0:
            return list(candidates.keys())[0], 0.0

        # Score each size
        scores = {}
        for size, candidate in candidates.items():
            expert_score = candidate.expert_count / total_experts
            mass_score = candidate.total_mass / total_mass
            # Weighted combination
            scores[size] = 0.4 * expert_score + 0.6 * mass_score

        best_size = max(scores.keys(), key=lambda s: scores[s])
        return best_size, scores[best_size]

    def _level2_content_scoring(
        self,
        candidates: Dict[Tuple[int, int], SizeCandidate]
    ) -> Dict[Tuple[int, int], float]:
        """
        Level 2: Score candidates based on grid content quality.

        Args:
            candidates: Size candidates

        Returns:
            Dict mapping size to content score
        """
        scores = {}

        for size, candidate in candidates.items():
            if not candidate.grids:
                scores[size] = 0.0
                continue

            # Compute agreement among grids of same size
            agreement = self._compute_grid_agreement(candidate.grids)

            # Combine with mass
            mass_score = candidate.avg_confidence
            scores[size] = 0.5 * agreement + 0.5 * mass_score

        return scores

    def _compute_grid_agreement(
        self,
        grids: List[Tuple[Grid, float, Dict]]
    ) -> float:
        """
        Compute agreement level among grids of the same size.

        Args:
            grids: List of (grid, mass, cell_masses)

        Returns:
            Agreement score in [0, 1]
        """
        if len(grids) <= 1:
            return 1.0

        # Extract just the grids
        grid_arrays = [g for g, _, _ in grids]

        # Cell-by-cell agreement
        h, w = grid_arrays[0].shape
        total_cells = h * w
        agreeing_cells = 0

        for i in range(h):
            for j in range(w):
                values = [g[i, j] for g in grid_arrays if g[i, j] >= 0]
                if values:
                    # Check if majority agrees
                    from collections import Counter
                    counts = Counter(values)
                    most_common = counts.most_common(1)[0][1]
                    if most_common >= len(values) * 0.5:
                        agreeing_cells += 1

        return agreeing_cells / total_cells if total_cells > 0 else 0.0

    def _level3_joint_scoring(
        self,
        content_scores: Dict[Tuple[int, int], float]
    ) -> Tuple[Tuple[int, int], float]:
        """
        Level 3: Joint scoring combining all factors.

        Args:
            content_scores: Content scores from level 2

        Returns:
            (best_size, joint_score)
        """
        if not content_scores:
            return (0, 0), 0.0

        best_size = max(content_scores.keys(), key=lambda s: content_scores[s])
        return best_size, content_scores[best_size]

    def resize_grids_to_target(
        self,
        grids: List[Tuple[Grid, float, Dict]],
        target_size: Tuple[int, int]
    ) -> List[Tuple[Grid, float, Dict]]:
        """
        Resize grids to target size (for combining different sizes).

        This is used when we want to combine grids of different sizes
        by padding/cropping to a common size.

        Args:
            grids: List of (grid, mass, cell_masses)
            target_size: Target (height, width)

        Returns:
            Resized grids with adjusted cell_masses
        """
        result = []
        target_h, target_w = target_size

        for grid, mass, cell_masses in grids:
            h, w = grid.shape

            if (h, w) == target_size:
                result.append((copy_grid(grid), mass, cell_masses.copy()))
                continue

            # Pad if smaller
            if h < target_h or w < target_w:
                new_grid = pad_grid(grid, target_h, target_w, fill_value=-1)
                # Update cell masses (original cells keep their masses)
                new_cell_masses = {
                    (i, j): m for (i, j), m in cell_masses.items()
                    if i < target_h and j < target_w
                }
                result.append((new_grid[:target_h, :target_w], mass * 0.8,
                               new_cell_masses))

            # Crop if larger
            elif h > target_h or w > target_w:
                new_grid = grid[:target_h, :target_w].copy()
                new_cell_masses = {
                    (i, j): m for (i, j), m in cell_masses.items()
                    if i < target_h and j < target_w
                }
                # Penalize mass for cropping
                result.append((new_grid, mass * 0.7, new_cell_masses))

            else:
                result.append((copy_grid(grid), mass, cell_masses.copy()))

        return result


def resolve_size_disagreement(
    expert_results: Dict[str, List[Tuple[Grid, float, Dict]]],
    config: Optional[MARCO4Config] = None
) -> SizeResolutionResult:
    """
    Convenience function to resolve size disagreements.

    Args:
        expert_results: Dict mapping expert_id to list of (grid, mass, cell_masses)
        config: Configuration (optional)

    Returns:
        SizeResolutionResult
    """
    resolver = SizeResolver(config)
    return resolver.resolve(expert_results)
