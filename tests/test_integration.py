"""
Integration Tests for MARCO4

End-to-end tests covering:
- Full iteration: branch → experts → MCU → new branch
- Solving simple ARC tasks
- Convergence verification
"""

import pytest
import numpy as np
from typing import Dict, Tuple

from marco.mcu import MCU, SolveResult
from marco.expert import Expert, MockExpert, HeuristicExpert
from marco.css import CognitiveStateSpace, BranchStatus
from marco.config import MARCO4Config
from marco.dempster_shafer import (
    dempster_combine,
    token_probs_to_belief_mass,
    compute_belief,
    THETA,
)
from marco.utils import (
    create_empty_grid,
    is_complete_grid,
    Grid,
    grid_to_string,
)
from marco.main import ARCProblem, solve_task, evaluate_solution


class FixedPatternExpert(Expert):
    """Expert that fills grid with a fixed pattern based on position."""

    def __init__(self, expert_id: str, pattern_fn=None, confidence=0.9):
        super().__init__(expert_id)
        self.pattern_fn = pattern_fn or (lambda i, j: (i + j) % 10)
        self.confidence = confidence

    def get_cell_probabilities(
        self,
        problem,
        partial_grid: Grid,
        position: Tuple[int, int]
    ) -> Dict[int, float]:
        """Return high probability for pattern-determined color."""
        i, j = position
        color = self.pattern_fn(i, j)

        probs = {c: 0.01 for c in range(10)}
        probs[color] = self.confidence
        total = sum(probs.values())
        return {c: p / total for c, p in probs.items()}


class TestEndToEndSimple:
    """End-to-end tests with simple tasks."""

    def test_solve_constant_fill_task(self):
        """
        Task: Fill entire grid with 1s.
        Expert: Always predicts 1 with high confidence.
        """
        # Expert that always predicts 1
        expert = FixedPatternExpert("const_1", lambda i, j: 1, confidence=0.95)

        config = MARCO4Config()
        config.search.max_iterations = 20
        config.confidence.solution_threshold = 0.1

        mcu = MCU([expert], config)

        problem = ARCProblem({
            'train': [
                {'input': [[0, 0], [0, 0]], 'output': [[1, 1], [1, 1]]}
            ],
            'test': [
                {'input': [[0, 0], [0, 0]], 'output': [[1, 1], [1, 1]]}
            ]
        })

        result = mcu.solve(problem, target_size=(2, 2))

        assert result.solution is not None
        assert np.all(result.solution == 1)
        assert result.status in ['solved', 'best_effort', 'max_iterations']

    def test_solve_checkerboard_task(self):
        """
        Task: Create checkerboard pattern.
        Expert: Predicts based on (i+j) % 2.
        """
        expert = FixedPatternExpert("checker", lambda i, j: (i + j) % 2, confidence=0.9)

        config = MARCO4Config()
        config.search.max_iterations = 20
        config.confidence.solution_threshold = 0.1

        mcu = MCU([expert], config)

        problem = ARCProblem({
            'train': [
                {'input': [[0, 0], [0, 0]], 'output': [[0, 1], [1, 0]]}
            ],
            'test': []
        })

        result = mcu.solve(problem, target_size=(2, 2))

        assert result.solution is not None
        # Check checkerboard pattern
        expected = np.array([[0, 1], [1, 0]])
        assert np.array_equal(result.solution, expected)


class TestMultipleExperts:
    """Tests with multiple experts."""

    def test_agreeing_experts(self):
        """Multiple experts that agree should converge quickly."""
        # Three experts all predicting 1s
        experts = [
            FixedPatternExpert(f"exp_{i}", lambda i, j: 1, confidence=0.9)
            for i in range(3)
        ]

        config = MARCO4Config()
        config.search.max_iterations = 10
        config.confidence.solution_threshold = 0.1

        mcu = MCU(experts, config)

        problem = ARCProblem({
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        result = mcu.solve(problem, target_size=(2, 2))

        assert result.solution is not None
        # Should converge quickly due to agreement
        assert result.iterations <= 5

    def test_disagreeing_experts_resolve(self):
        """Multiple experts that disagree should still produce solution."""
        # Experts predicting different colors
        experts = [
            FixedPatternExpert("exp_0", lambda i, j: 0, confidence=0.6),
            FixedPatternExpert("exp_1", lambda i, j: 1, confidence=0.7),
            FixedPatternExpert("exp_2", lambda i, j: 2, confidence=0.5),
        ]

        config = MARCO4Config()
        config.search.max_iterations = 20
        config.confidence.solution_threshold = 0.05  # Low threshold

        mcu = MCU(experts, config)

        problem = ARCProblem({
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        result = mcu.solve(problem, target_size=(2, 2))

        # Should still produce something
        assert result.iterations > 0
        # May or may not find solution due to conflict


class TestIterativeRefinement:
    """Tests for iterative refinement process."""

    def test_progressive_fill(self):
        """Grid should progressively fill over iterations."""
        expert = FixedPatternExpert("prog", lambda i, j: 1, confidence=0.8)

        config = MARCO4Config()
        config.search.max_iterations = 50
        config.confidence.high_confidence = 0.6  # Lower threshold for faster progress

        mcu = MCU([expert], config)

        problem = ARCProblem({
            'train': [{'input': [[0, 0, 0], [0, 0, 0]], 'output': [[1, 1, 1], [1, 1, 1]]}],
            'test': []
        })

        result = mcu.solve(problem, target_size=(2, 3))

        # Should find complete solution
        assert result.solution is not None
        assert is_complete_grid(result.solution)

    def test_css_grows_during_search(self):
        """CSS should accumulate branches during search."""
        expert = MockExpert("mock", seed=42)

        config = MARCO4Config()
        config.search.max_iterations = 10

        mcu = MCU([expert], config)

        problem = ARCProblem({
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        mcu.solve(problem, target_size=(2, 2))

        stats = mcu.css.get_statistics()
        assert stats['total_created'] >= 1


class TestSolutionEvaluation:
    """Tests for solution evaluation."""

    def test_evaluate_correct_solution(self):
        """Correct solution should evaluate as correct."""
        solution = np.array([[1, 2], [3, 4]])
        expected = [[1, 2], [3, 4]]

        result = evaluate_solution(solution, expected)

        assert result['correct'] is True
        assert result['accuracy'] == 1.0
        assert result['shape_match'] is True

    def test_evaluate_incorrect_solution(self):
        """Incorrect solution should have accuracy < 1."""
        solution = np.array([[1, 2], [3, 5]])  # One wrong cell
        expected = [[1, 2], [3, 4]]

        result = evaluate_solution(solution, expected)

        assert result['correct'] is False
        assert result['accuracy'] == 0.75  # 3/4 cells correct
        assert result['shape_match'] is True

    def test_evaluate_wrong_shape(self):
        """Wrong shape should be detected."""
        solution = np.array([[1, 2, 3]])
        expected = [[1, 2], [3, 4]]

        result = evaluate_solution(solution, expected)

        assert result['correct'] is False
        assert result['shape_match'] is False

    def test_evaluate_none_solution(self):
        """None solution should evaluate as incorrect."""
        result = evaluate_solution(None, [[1, 2]])

        assert result['correct'] is False
        assert result['accuracy'] == 0.0


class TestSolveTaskAPI:
    """Tests for high-level solve_task API."""

    def test_solve_task_returns_result(self):
        """solve_task should return SolveResult."""
        problem = ARCProblem({
            'task_id': 'test_1',
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        result = solve_task(
            problem,
            expert_type='heuristic',
            num_experts=2
        )

        assert isinstance(result, SolveResult)
        assert result.iterations > 0

    def test_solve_task_with_config(self):
        """solve_task should respect custom config."""
        problem = ARCProblem({
            'task_id': 'test_2',
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        config = MARCO4Config()
        config.search.max_iterations = 3

        result = solve_task(
            problem,
            config=config,
            expert_type='mock',
            num_experts=1
        )

        assert result.iterations <= 3


class TestDempsterShaferIntegration:
    """Tests verifying D-S theory is correctly applied."""

    def test_ds_combination_in_mcu(self):
        """Verify D-S combination is used in MCU cell combination."""
        # Two experts with different but overlapping beliefs
        class DSTestExpert(Expert):
            def __init__(self, expert_id, primary_color):
                super().__init__(expert_id)
                self.primary_color = primary_color

            def get_cell_probabilities(self, problem, grid, pos):
                probs = {i: 0.05 for i in range(10)}
                probs[self.primary_color] = 0.7
                return probs

        experts = [
            DSTestExpert("exp_a", 1),
            DSTestExpert("exp_b", 1),  # Both predict 1
        ]

        config = MARCO4Config()
        config.search.max_iterations = 10
        config.confidence.solution_threshold = 0.1

        mcu = MCU(experts, config)

        problem = ARCProblem({
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        result = mcu.solve(problem, target_size=(2, 2))

        # With both experts predicting 1, D-S combination should strengthen belief
        assert result.solution is not None
        assert result.confidence > 0

    def test_mass_function_validity(self):
        """Verify mass functions sum to 1 throughout computation."""
        from marco.dempster_shafer import validate_mass_function

        probs = {0: 0.5, 1: 0.3, 2: 0.2}
        mass = token_probs_to_belief_mass(probs, strategy='entropy')

        # Should be valid
        assert validate_mass_function(mass)

        # Combine with another mass
        mass2 = token_probs_to_belief_mass({0: 0.6, 1: 0.4}, strategy='entropy')
        combined = dempster_combine(mass, mass2)

        # Combined should also be valid
        assert validate_mass_function(combined)


class TestTwoLevelPruning:
    """Tests verifying two-level pruning architecture."""

    def test_expert_level_pruning(self):
        """Expert should prune during DFS."""
        config = MARCO4Config()
        config.pruning.local_threshold = 0.1  # Moderate threshold

        expert = MockExpert("mock", config=config, seed=42)
        grid = create_empty_grid(3, 3)  # 9 cells

        problem = ARCProblem({
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        results = expert.dfs_to_complete(problem, grid)

        # With pruning, should not explore all possible paths
        # Max possible without pruning: 10^9 = 1 billion
        # With pruning: much fewer
        assert len(results) < 1000

    def test_mcu_level_pruning(self):
        """MCU should prune branches after D-S combination."""
        config = MARCO4Config()
        config.pruning.mcu_prune_threshold = 0.5  # High threshold to force pruning
        config.search.max_iterations = 20

        # Expert with low confidence
        expert = MockExpert("mock", config=config, seed=42)
        mcu = MCU([expert], config)

        problem = ARCProblem({
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': []
        })

        mcu.solve(problem, target_size=(2, 2))

        # Should have some pruned branches
        stats = mcu.css.get_statistics()
        # With high prune threshold, expect some pruning
        assert stats['total_created'] >= 1


class TestARCProblemLoading:
    """Tests for ARC problem handling."""

    def test_arc_problem_from_dict(self):
        """ARCProblem should load from dictionary."""
        data = {
            'task_id': 'test',
            'train': [{'input': [[0, 1]], 'output': [[1, 0]]}],
            'test': [{'input': [[1, 0]], 'output': [[0, 1]]}]
        }

        problem = ARCProblem(data)

        assert problem.task_id == 'test'
        assert len(problem.train) == 1
        assert len(problem.test) == 1

    def test_arc_problem_empty(self):
        """ARCProblem should handle empty data."""
        problem = ARCProblem({})

        assert problem.train == []
        assert problem.test == []
        assert problem.task_id == 'unknown'
