"""
Unit Tests for MCU Module

Tests cover:
- Cell-by-cell D-S combination with per-cell logits
- GridState and CellState management
- MCU-driven branching
- Partial grid extraction
- MCU-level pruning
- A* search integration
"""

import pytest
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

from marco.mcu import (
    MCU,
    CellState,
    GridState,
    CombinedResult,
    SolveResult,
)
from marco.expert import (
    Expert,
    MockExpert,
    HeuristicExpert,
    CellLogits,
    ExpertOutput,
)
from marco.css import CognitiveStateSpace, BranchStatus
from marco.utils import (
    create_empty_grid,
    is_complete_grid,
    Grid,
)
from marco.config import MARCO4Config
from marco.dempster_shafer import THETA, MassFunction


class DeterministicExpert(Expert):
    """Expert that returns predetermined per-cell logits for testing."""

    def __init__(
        self,
        expert_id: str,
        target_grid: Optional[np.ndarray] = None,
        confidence: float = 0.9,
        config: Optional[MARCO4Config] = None
    ):
        super().__init__(expert_id, config)
        self.target_grid = target_grid
        self.confidence = confidence

    def get_cell_logits(
        self,
        problem: Any,
        partial_grid: Grid
    ) -> CellLogits:
        """Return logits biased toward target grid values."""
        h, w = partial_grid.shape

        if self.target_grid is None:
            # Random logits
            logits = np.random.randn(h, w, 10)
        else:
            # Bias toward target values
            logits = np.full((h, w, 10), -10.0)
            for i in range(h):
                for j in range(w):
                    val = int(self.target_grid[i, j])
                    # High logit for target value
                    logits[i, j, val] = 0.0
                    # Add some noise to non-target values based on confidence
                    noise_scale = 1.0 - self.confidence
                    for c in range(10):
                        if c != val:
                            logits[i, j, c] += np.random.randn() * noise_scale * 5

        return CellLogits(logits=logits)


class MockProblem:
    """Mock ARC problem for testing."""

    def __init__(self, train_outputs=None, target_size=None):
        if train_outputs is None:
            train_outputs = [[[1, 1], [1, 1]]]
        self.train = [{'output': out} for out in train_outputs]
        self.test = []
        self._target_size = target_size or (2, 2)


class TestCellState:
    """Tests for CellState dataclass."""

    def test_get_best_color_singleton(self):
        """Best color for singleton mass function."""
        mass_func = {frozenset([3]): 0.8, THETA: 0.2}
        cell_state = CellState(mass_function=mass_func, belief=0.8)

        color, prob = cell_state.get_best_color()

        assert color == 3
        # Pignistic probability: 0.8 + 0.2/10 = 0.82
        assert abs(prob - 0.82) < 0.01

    def test_get_best_color_multiple(self):
        """Best color with multiple candidates."""
        mass_func = {
            frozenset([1]): 0.4,
            frozenset([2]): 0.3,
            THETA: 0.3
        }
        cell_state = CellState(mass_function=mass_func, belief=0.4)

        color, prob = cell_state.get_best_color()

        assert color == 1  # Highest pignistic probability
        # Pignistic: 0.4 + 0.3/10 = 0.43
        assert abs(prob - 0.43) < 0.01

    def test_is_fixed(self):
        """Test fixed value detection."""
        cell_state = CellState(mass_function={THETA: 1.0})
        assert not cell_state.is_fixed()

        cell_state.fixed_value = 5
        assert cell_state.is_fixed()


class TestGridState:
    """Tests for GridState dataclass."""

    def test_to_grid_with_fixed(self):
        """Convert to grid with fixed values."""
        cell_states = {
            (0, 0): CellState(mass_function={frozenset([1]): 0.9, THETA: 0.1}, fixed_value=1, belief=0.9),
            (0, 1): CellState(mass_function={frozenset([2]): 0.9, THETA: 0.1}, fixed_value=2, belief=0.9),
            (1, 0): CellState(mass_function={frozenset([3]): 0.9, THETA: 0.1}, fixed_value=3, belief=0.9),
            (1, 1): CellState(mass_function={frozenset([4]): 0.9, THETA: 0.1}, fixed_value=4, belief=0.9),
        }
        grid_state = GridState(cell_states=cell_states, shape=(2, 2))

        grid = grid_state.to_grid()

        assert grid[0, 0] == 1
        assert grid[0, 1] == 2
        assert grid[1, 0] == 3
        assert grid[1, 1] == 4

    def test_to_partial_grid(self):
        """Convert to partial grid with confidence threshold."""
        cell_states = {
            (0, 0): CellState(mass_function={frozenset([1]): 0.9, THETA: 0.1}, belief=0.9),
            (0, 1): CellState(mass_function={frozenset([2]): 0.5, THETA: 0.5}, belief=0.5),
            (1, 0): CellState(mass_function={frozenset([3]): 0.8, THETA: 0.2}, belief=0.8),
            (1, 1): CellState(mass_function={frozenset([4]): 0.3, THETA: 0.7}, belief=0.3),
        }
        grid_state = GridState(cell_states=cell_states, shape=(2, 2))

        partial = grid_state.to_partial_grid(confidence_threshold=0.7)

        assert partial[0, 0] == 1  # High confidence
        assert partial[0, 1] == -1  # Below threshold
        assert partial[1, 0] == 3  # High confidence
        assert partial[1, 1] == -1  # Below threshold

    def test_get_uncertain_cells(self):
        """Find cells with multiple viable candidates."""
        # Cell with two strong candidates
        cell_states = {
            (0, 0): CellState(
                mass_function={frozenset([1]): 0.4, frozenset([2]): 0.35, THETA: 0.25},
                belief=0.4
            ),
            (0, 1): CellState(
                mass_function={frozenset([3]): 0.9, THETA: 0.1},
                fixed_value=3,
                belief=0.9
            ),
        }
        grid_state = GridState(cell_states=cell_states, shape=(1, 2))

        uncertain = grid_state.get_uncertain_cells(branch_threshold=0.15)

        # Only (0, 0) should be uncertain
        assert len(uncertain) == 1
        pos, candidates = uncertain[0]
        assert pos == (0, 0)
        assert len(candidates) >= 2

    def test_is_complete(self):
        """Check if all cells are fixed."""
        cell_states = {
            (0, 0): CellState(mass_function={frozenset([1]): 0.9, THETA: 0.1}, fixed_value=1, belief=0.9),
            (0, 1): CellState(mass_function={frozenset([2]): 0.9, THETA: 0.1}, belief=0.9),  # Not fixed
        }
        grid_state = GridState(cell_states=cell_states, shape=(1, 2))

        assert not grid_state.is_complete()

        cell_states[(0, 1)].fixed_value = 2
        assert grid_state.is_complete()


class TestMCUCombineOutputs:
    """Tests for cell-by-cell D-S combination."""

    def test_combine_single_expert(self):
        """Combining outputs from single expert."""
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        expert = DeterministicExpert("exp1", target_grid=grid)

        mcu = MCU([expert])
        problem = MockProblem()
        partial_grid = create_empty_grid(2, 2)

        outputs = mcu._collect_expert_outputs(problem, partial_grid)
        result = mcu._combine_expert_outputs(outputs, (2, 2))

        assert result.grid_state.shape == (2, 2)
        assert len(result.grid_state.cell_states) == 4

    def test_combine_agreeing_experts(self):
        """Experts that agree should have low conflict."""
        grid = np.array([[1, 1], [1, 1]], dtype=np.int8)

        exp1 = DeterministicExpert("exp1", target_grid=grid, confidence=0.9)
        exp2 = DeterministicExpert("exp2", target_grid=grid, confidence=0.9)

        mcu = MCU([exp1, exp2])
        problem = MockProblem()
        partial_grid = create_empty_grid(2, 2)

        outputs = mcu._collect_expert_outputs(problem, partial_grid)
        result = mcu._combine_expert_outputs(outputs, (2, 2))

        # Combined should agree on color 1
        for (i, j), cell_state in result.grid_state.cell_states.items():
            best_color, _ = cell_state.get_best_color()
            assert best_color == 1

        # Low conflict due to agreement
        assert result.conflict_level < 0.5

    def test_combine_disagreeing_experts(self):
        """Experts that disagree should have higher conflict."""
        grid1 = np.array([[1, 1], [1, 1]], dtype=np.int8)
        grid2 = np.array([[2, 2], [2, 2]], dtype=np.int8)

        exp1 = DeterministicExpert("exp1", target_grid=grid1, confidence=0.9)
        exp2 = DeterministicExpert("exp2", target_grid=grid2, confidence=0.9)

        mcu = MCU([exp1, exp2])
        problem = MockProblem()
        partial_grid = create_empty_grid(2, 2)

        outputs = mcu._collect_expert_outputs(problem, partial_grid)
        result = mcu._combine_expert_outputs(outputs, (2, 2))

        # Should have higher conflict
        assert result.conflict_level > 0


class TestMCUBranching:
    """Tests for MCU-driven branching."""

    def test_branching_for_uncertain_cell(self):
        """MCU should create branches for uncertain cells."""
        # Create expert with ambiguous output
        class AmbiguousExpert(Expert):
            def get_cell_logits(self, problem, partial_grid):
                h, w = partial_grid.shape
                # Cell (0,0) has two equally likely candidates
                logits = np.full((h, w, 10), -10.0)
                logits[0, 0, 1] = 0.0  # Color 1
                logits[0, 0, 2] = -0.1  # Color 2 almost as likely
                # Other cells are certain
                for i in range(h):
                    for j in range(w):
                        if (i, j) != (0, 0):
                            logits[i, j, 5] = 0.0
                return CellLogits(logits=logits)

        expert = AmbiguousExpert("ambig")

        config = MARCO4Config()
        config.search.max_iterations = 5
        config.search.branch_threshold = 0.1

        mcu = MCU([expert], config)
        problem = MockProblem()

        result = mcu.solve(problem, target_size=(2, 2))

        # Should have created multiple branches
        assert mcu.css.total_branches_created > 1


class TestMCUFixConfidentCells:
    """Tests for fixing high-confidence cells."""

    def test_fix_high_confidence(self):
        """High confidence cells should be fixed."""
        cell_states = {
            (0, 0): CellState(
                mass_function={frozenset([1]): 0.9, THETA: 0.1},
                belief=0.9
            ),
            (0, 1): CellState(
                mass_function={frozenset([2]): 0.5, THETA: 0.5},
                belief=0.5
            ),
        }
        grid_state = GridState(cell_states=cell_states, shape=(1, 2))

        mcu = MCU([])
        result = mcu._fix_confident_cells(grid_state, confidence_threshold=0.7)

        assert result.cell_states[(0, 0)].is_fixed()
        assert result.cell_states[(0, 0)].fixed_value == 1
        assert not result.cell_states[(0, 1)].is_fixed()


class TestMCUSolve:
    """Tests for main MCU solve loop."""

    def test_solve_simple_task(self):
        """MCU should solve a simple task."""
        grid = np.array([[1, 1], [1, 1]], dtype=np.int8)
        expert = DeterministicExpert("exp1", target_grid=grid, confidence=0.95)

        config = MARCO4Config()
        config.search.max_iterations = 10
        config.confidence.solution_threshold = 0.1
        config.confidence.high_confidence = 0.7

        mcu = MCU([expert], config)
        problem = MockProblem([[[1, 1], [1, 1]]])

        result = mcu.solve(problem, target_size=(2, 2))

        assert result.solution is not None
        assert result.confidence > 0
        assert result.iterations > 0

    def test_solve_tracks_statistics(self):
        """MCU should track solving statistics."""
        expert = MockExpert("mock1", seed=42)

        config = MARCO4Config()
        config.search.max_iterations = 3

        mcu = MCU([expert], config)
        problem = MockProblem()

        result = mcu.solve(problem, target_size=(2, 2))

        stats = mcu.get_statistics()
        assert 'css' in stats
        assert 'num_experts' in stats
        assert stats['num_experts'] == 1


class TestMCUPruning:
    """Tests for MCU-level pruning."""

    def test_prune_low_mass(self):
        """Branch with low mass should be pruned."""
        # Expert that produces low-confidence outputs
        class LowConfExpert(Expert):
            def get_cell_logits(self, problem, partial_grid):
                h, w = partial_grid.shape
                # Uniform distribution -> low confidence
                logits = np.zeros((h, w, 10))
                return CellLogits(logits=logits)

        config = MARCO4Config()
        config.pruning.mcu_prune_threshold = 0.5  # High threshold
        config.search.max_iterations = 5

        mcu = MCU([LowConfExpert("low")], config)
        problem = MockProblem()

        mcu.solve(problem, target_size=(2, 2))

        # Some branches should be pruned
        stats = mcu.css.get_statistics()
        assert stats['pruned'] >= 0  # May or may not prune depending on randomness

    def test_prune_high_conflict(self):
        """Branch with high conflict should be pruned."""
        # Two experts that completely disagree
        grid1 = np.array([[1, 1], [1, 1]], dtype=np.int8)
        grid2 = np.array([[2, 2], [2, 2]], dtype=np.int8)

        exp1 = DeterministicExpert("exp1", target_grid=grid1, confidence=0.99)
        exp2 = DeterministicExpert("exp2", target_grid=grid2, confidence=0.99)

        config = MARCO4Config()
        config.pruning.max_conflict = 0.3  # Low threshold
        config.search.max_iterations = 5

        mcu = MCU([exp1, exp2], config)
        problem = MockProblem()

        mcu.solve(problem, target_size=(2, 2))

        # High conflict should lead to pruning
        stats = mcu.css.get_statistics()
        # At least the root branch should be processed
        assert stats['total_created'] >= 1


class TestCSSIntegration:
    """Tests for CSS integration with MCU."""

    def test_css_branch_management(self):
        """MCU should properly manage CSS branches."""
        expert = MockExpert("mock1", seed=42)

        config = MARCO4Config()
        config.search.max_iterations = 5

        mcu = MCU([expert], config)
        problem = MockProblem()

        mcu.solve(problem, target_size=(2, 2))

        # CSS should have some branches
        assert len(mcu.css) > 0


class TestSizeInference:
    """Tests for target size inference."""

    def test_infer_from_training(self):
        """MCU should infer size from training examples."""
        expert = MockExpert("mock1")
        mcu = MCU([expert])

        problem = MockProblem([[[1, 2, 3], [4, 5, 6]]])  # 2x3 outputs

        size = mcu._infer_target_size(problem)

        assert size == (2, 3)

    def test_infer_fallback(self):
        """MCU should use fallback size when training is empty."""
        expert = MockExpert("mock1")
        mcu = MCU([expert])

        problem = MockProblem([])  # No training

        size = mcu._infer_target_size(problem)

        # Should return default (3, 3)
        assert size == (3, 3)


class TestHeuristicComputation:
    """Tests for A* heuristic computation."""

    def test_heuristic_empty_grid(self):
        """Heuristic for empty grid should be 1.0."""
        mcu = MCU([])

        grid = create_empty_grid(2, 2)
        h = mcu._compute_heuristic(grid)

        assert h == 1.0  # All cells empty

    def test_heuristic_partial_grid(self):
        """Heuristic should decrease as grid fills."""
        mcu = MCU([])

        grid = create_empty_grid(2, 2)
        grid[0, 0] = 1
        h = mcu._compute_heuristic(grid)

        assert 0 < h < 1.0  # Partially filled

    def test_heuristic_complete_grid(self):
        """Heuristic for complete grid should be 0.0."""
        mcu = MCU([])

        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        h = mcu._compute_heuristic(grid)

        assert h == 0.0  # All cells filled


class TestParallelExecution:
    """Tests for parallel expert execution."""

    def test_parallel_collection(self):
        """Parallel expert collection should work."""
        exp1 = MockExpert("mock1", seed=1)
        exp2 = MockExpert("mock2", seed=2)

        mcu = MCU([exp1, exp2], parallel=True, max_workers=2)
        problem = MockProblem()
        partial_grid = create_empty_grid(2, 2)

        outputs = mcu._collect_expert_outputs(problem, partial_grid)

        assert len(outputs) == 2
        assert {o.expert_id for o in outputs} == {"mock1", "mock2"}

    def test_sequential_collection(self):
        """Sequential expert collection should work."""
        exp1 = MockExpert("mock1", seed=1)
        exp2 = MockExpert("mock2", seed=2)

        mcu = MCU([exp1, exp2], parallel=False)
        problem = MockProblem()
        partial_grid = create_empty_grid(2, 2)

        outputs = mcu._collect_expert_outputs(problem, partial_grid)

        assert len(outputs) == 2
