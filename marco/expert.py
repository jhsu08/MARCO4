"""
Expert Module for MARCO4

Implements the Expert class that outputs per-cell logits (H, W, 10)
for each cell in the grid, enabling proper D-S combination.

Each expert outputs a probability distribution over colors 0-9 for each cell.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import Grid, copy_grid, is_complete_grid
from .dempster_shafer import MassFunction, THETA, token_probs_to_belief_mass
from .config import MARCO4Config, DEFAULT_CONFIG


@dataclass
class CellLogits:
    """Per-cell logits output from an expert."""
    logits: np.ndarray  # Shape (H, W, 10) - log probabilities for each color

    @property
    def shape(self) -> Tuple[int, int]:
        return self.logits.shape[:2]

    def get_probs(self, row: int, col: int) -> np.ndarray:
        """Get probability distribution for a specific cell."""
        # Convert log-probs to probs via softmax
        log_probs = self.logits[row, col]
        max_lp = np.max(log_probs)
        probs = np.exp(log_probs - max_lp)
        return probs / probs.sum()

    def get_log_probs(self, row: int, col: int) -> np.ndarray:
        """Get log probabilities for a specific cell."""
        return self.logits[row, col]

    def to_mass_functions(self) -> Dict[Tuple[int, int], MassFunction]:
        """Convert all cells to D-S mass functions."""
        h, w = self.shape
        masses = {}
        for i in range(h):
            for j in range(w):
                probs = self.get_probs(i, j)
                prob_dict = {c: float(probs[c]) for c in range(10)}
                masses[(i, j)] = token_probs_to_belief_mass(prob_dict)
        return masses


@dataclass
class ExpertOutput:
    """Output from an expert."""
    expert_id: str
    cell_logits: CellLogits  # Per-cell logits (H, W, 10)

    def get_mass_functions(self) -> Dict[Tuple[int, int], MassFunction]:
        """Convert to per-cell mass functions for D-S combination."""
        return self.cell_logits.to_mass_functions()


class Expert(ABC):
    """
    Abstract base class for MARCO4 experts.

    Experts output per-cell logits: a (H, W, 10) tensor where each cell
    has log-probabilities for colors 0-9.

    This enables:
    - Proper D-S combination at the cell level
    - Multiple candidate values per cell before pruning
    - Uncertainty quantification per cell
    """

    def __init__(
        self,
        expert_id: str,
        config: Optional[MARCO4Config] = None
    ):
        self.expert_id = expert_id
        self.config = config or DEFAULT_CONFIG

    @abstractmethod
    def get_cell_logits(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> CellLogits:
        """
        Get per-cell logits for the grid.

        Args:
            problem: ARC problem data
            partial_grid: Current partial grid (-1 for empty cells)

        Returns:
            CellLogits with shape (H, W, 10) log-probabilities
        """
        pass

    def get_output(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> ExpertOutput:
        """
        Get expert output for MCU consumption.

        Args:
            problem: ARC problem data
            partial_grid: Current partial grid

        Returns:
            ExpertOutput with per-cell logits
        """
        cell_logits = self.get_cell_logits(problem, partial_grid)
        return ExpertOutput(
            expert_id=self.expert_id,
            cell_logits=cell_logits
        )


class MockExpert(Expert):
    """Mock expert for testing that generates random per-cell logits."""

    def __init__(
        self,
        expert_id: str,
        config: Optional[MARCO4Config] = None,
        seed: Optional[int] = None
    ):
        super().__init__(expert_id, config)
        self.rng = np.random.RandomState(seed)

    def get_cell_logits(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> CellLogits:
        """Generate random per-cell logits for testing."""
        h, w = partial_grid.shape

        # Random logits for each cell
        logits = self.rng.randn(h, w, 10)

        # For already-filled cells, strongly bias toward the existing value
        for i in range(h):
            for j in range(w):
                if partial_grid[i, j] >= 0:
                    val = int(partial_grid[i, j])
                    logits[i, j, :] = -10.0  # Low prob for all
                    logits[i, j, val] = 0.0   # High prob for existing value

        return CellLogits(logits=logits)


class HeuristicExpert(Expert):
    """Expert that uses heuristics based on training examples."""

    def __init__(
        self,
        expert_id: str,
        config: Optional[MARCO4Config] = None
    ):
        super().__init__(expert_id, config)
        self._color_log_probs: Optional[np.ndarray] = None

    def _analyze_training(self, problem: Any) -> np.ndarray:
        """Analyze training to get color log-probabilities."""
        if self._color_log_probs is not None:
            return self._color_log_probs

        color_counts = np.ones(10)  # Laplace smoothing

        if hasattr(problem, 'train') and problem.train:
            for example in problem.train:
                output = example.get('output', [])
                if output:
                    arr = np.array(output)
                    for val in arr.flatten():
                        if 0 <= val <= 9:
                            color_counts[int(val)] += 1

        # Convert to log probabilities
        probs = color_counts / color_counts.sum()
        self._color_log_probs = np.log(probs + 1e-10)
        return self._color_log_probs

    def get_cell_logits(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> CellLogits:
        """Generate per-cell logits based on color frequencies."""
        base_log_probs = self._analyze_training(problem)
        h, w = partial_grid.shape

        # Start with base distribution for all cells
        logits = np.tile(base_log_probs, (h, w, 1))

        # For filled cells, strongly bias toward existing value
        for i in range(h):
            for j in range(w):
                if partial_grid[i, j] >= 0:
                    val = int(partial_grid[i, j])
                    logits[i, j, :] = -10.0
                    logits[i, j, val] = 0.0
                else:
                    # Boost based on neighbors
                    neighbor_vals = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and partial_grid[ni, nj] >= 0:
                            neighbor_vals.append(int(partial_grid[ni, nj]))

                    # Boost probability of neighbor colors
                    for val in neighbor_vals:
                        logits[i, j, val] += 1.0

        return CellLogits(logits=logits)
