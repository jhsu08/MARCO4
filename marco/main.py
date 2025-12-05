"""
Main Entry Point for MARCO4

Provides command-line interface and programmatic API for solving ARC tasks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from .mcu import MCU, SolveResult
from .expert import Expert, HeuristicExpert, MockExpert
from .config import MARCO4Config
from .utils import grid_to_string


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARCProblem:
    """Simple wrapper for ARC problem data."""

    def __init__(self, data: Dict[str, Any]):
        self.train = data.get('train', [])
        self.test = data.get('test', [])
        self.task_id = data.get('task_id', 'unknown')

    @classmethod
    def from_file(cls, path: str) -> 'ARCProblem':
        """Load problem from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data)


def create_experts(
    expert_type: str = 'heuristic',
    num_experts: int = 3,
    config: Optional[MARCO4Config] = None
) -> List[Expert]:
    """
    Create expert instances.

    Args:
        expert_type: Type of experts ('heuristic', 'mock', 'llm')
        num_experts: Number of experts to create
        config: Configuration

    Returns:
        List of Expert instances
    """
    experts = []

    for i in range(num_experts):
        expert_id = f"{expert_type}_{i}"

        if expert_type == 'heuristic':
            expert = HeuristicExpert(expert_id, config)
        elif expert_type == 'mock':
            expert = MockExpert(expert_id, config, seed=i * 42)
        else:
            # Default to heuristic
            expert = HeuristicExpert(expert_id, config)

        experts.append(expert)

    return experts


def solve_task(
    problem: ARCProblem,
    config: Optional[MARCO4Config] = None,
    expert_type: str = 'heuristic',
    num_experts: int = 3,
    verbose: bool = False
) -> SolveResult:
    """
    Solve an ARC task using MARCO4.

    Args:
        problem: ARC problem
        config: Configuration (uses defaults if None)
        expert_type: Type of experts to use
        num_experts: Number of experts
        verbose: Enable verbose logging

    Returns:
        SolveResult with solution
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = config or MARCO4Config()
    experts = create_experts(expert_type, num_experts, config)

    mcu = MCU(experts, config)

    # Infer target size from training examples
    target_size = None
    if problem.train:
        output = problem.train[0].get('output', [])
        if output:
            target_size = (len(output), len(output[0]) if output else 0)

    logger.info(f"Solving task {problem.task_id} with target size {target_size}")

    result = mcu.solve(problem, target_size)

    logger.info(f"Result: {result.status}, confidence={result.confidence:.4f}, "
                f"iterations={result.iterations}, branches={result.branches_explored}")

    return result


def evaluate_solution(
    solution: np.ndarray,
    expected: List[List[int]]
) -> Dict[str, Any]:
    """
    Evaluate solution against expected output.

    Args:
        solution: Predicted grid
        expected: Expected output grid

    Returns:
        Evaluation metrics
    """
    expected_arr = np.array(expected)

    if solution is None:
        return {
            'correct': False,
            'accuracy': 0.0,
            'shape_match': False
        }

    shape_match = solution.shape == expected_arr.shape

    if not shape_match:
        return {
            'correct': False,
            'accuracy': 0.0,
            'shape_match': False
        }

    correct = np.array_equal(solution, expected_arr)
    accuracy = np.mean(solution == expected_arr)

    return {
        'correct': correct,
        'accuracy': float(accuracy),
        'shape_match': True
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='MARCO4: Multi-Agent Reasoning for ARC'
    )

    parser.add_argument(
        'task_file',
        type=str,
        nargs='?',
        help='Path to ARC task JSON file'
    )

    parser.add_argument(
        '--expert-type',
        type=str,
        default='heuristic',
        choices=['heuristic', 'mock'],
        help='Type of experts to use'
    )

    parser.add_argument(
        '--num-experts',
        type=int,
        default=3,
        help='Number of experts'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help='Maximum iterations'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with synthetic task'
    )

    args = parser.parse_args()

    if args.demo:
        # Run demo with synthetic task
        demo_task = {
            'task_id': 'demo_1',
            'train': [
                {
                    'input': [[0, 0], [0, 0]],
                    'output': [[1, 1], [1, 1]]
                },
                {
                    'input': [[0, 0, 0], [0, 0, 0]],
                    'output': [[1, 1, 1], [1, 1, 1]]
                }
            ],
            'test': [
                {
                    'input': [[0, 0], [0, 0], [0, 0]],
                    'output': [[1, 1], [1, 1], [1, 1]]
                }
            ]
        }

        problem = ARCProblem(demo_task)

        config = MARCO4Config()
        config.search.max_iterations = args.max_iterations

        result = solve_task(
            problem,
            config=config,
            expert_type=args.expert_type,
            num_experts=args.num_experts,
            verbose=args.verbose
        )

        print("\n" + "=" * 50)
        print("DEMO RESULTS")
        print("=" * 50)
        print(f"Status: {result.status}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Iterations: {result.iterations}")
        print(f"Branches explored: {result.branches_explored}")

        if result.solution is not None:
            print("\nSolution:")
            print(grid_to_string(result.solution))

            # Evaluate
            expected = demo_task['test'][0]['output']
            eval_result = evaluate_solution(result.solution, expected)
            print(f"\nCorrect: {eval_result['correct']}")
            print(f"Accuracy: {eval_result['accuracy']:.2%}")

        return 0

    if args.task_file is None:
        parser.print_help()
        return 1

    # Load and solve task
    try:
        problem = ARCProblem.from_file(args.task_file)
    except FileNotFoundError:
        print(f"Error: Task file not found: {args.task_file}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in task file: {e}")
        return 1

    config = MARCO4Config()
    config.search.max_iterations = args.max_iterations

    result = solve_task(
        problem,
        config=config,
        expert_type=args.expert_type,
        num_experts=args.num_experts,
        verbose=args.verbose
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Task: {problem.task_id}")
    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Iterations: {result.iterations}")
    print(f"Branches explored: {result.branches_explored}")

    if result.solution is not None:
        print("\nSolution:")
        print(grid_to_string(result.solution))

        # Evaluate if test output available
        if problem.test and 'output' in problem.test[0]:
            expected = problem.test[0]['output']
            eval_result = evaluate_solution(result.solution, expected)
            print(f"\nCorrect: {eval_result['correct']}")
            print(f"Accuracy: {eval_result['accuracy']:.2%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
