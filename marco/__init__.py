"""
MARCO4: Meta-cognitive Architecture for Reasoning and Contextual Oversight

A two-level hierarchical architecture where experts score complete candidate grids
and the MCU combines them using Dempster-Shafer theory.

Key Features:
- Dempster-Shafer theory for evidence combination
- Experts score complete grids using sequence-level log probabilities
- PoE (Product of Experts) via augmentations
- A* search through Cognitive State Space
- Cell-by-cell D-S combination by MCU
"""

from .dempster_shafer import (
    dempster_combine,
    dempster_combine_multiple,
    compute_belief,
    compute_plausibility,
    compute_pignistic_probability,
    get_pignistic_distribution,
    token_probs_to_belief_mass,
    get_best_color,
    get_conflict_level,
    THETA,
    MassFunction,
)
from .config import DEFAULT_CONFIG, MARCO4Config
from .css import CognitiveStateSpace, BranchNode, BranchStatus
from .expert import Expert, MockExpert, HeuristicExpert, CellLogits, ExpertOutput
from .mcu import MCU, SolveResult, CombinedResult
from .size_resolution import SizeResolver, resolve_size_disagreement
from .utils import (
    create_empty_grid,
    is_complete_grid,
    is_partial_grid,
    grid_to_string,
    grid_to_hash,
    apply_augmentation,
)
from .main import ARCProblem, solve_task, evaluate_solution

__version__ = "4.0.0"
__all__ = [
    # Dempster-Shafer
    "dempster_combine",
    "dempster_combine_multiple",
    "compute_belief",
    "compute_plausibility",
    "compute_pignistic_probability",
    "get_pignistic_distribution",
    "token_probs_to_belief_mass",
    "get_best_color",
    "get_conflict_level",
    "THETA",
    "MassFunction",
    # Config
    "DEFAULT_CONFIG",
    "MARCO4Config",
    # CSS
    "CognitiveStateSpace",
    "BranchNode",
    "BranchStatus",
    # Experts
    "Expert",
    "MockExpert",
    "HeuristicExpert",
    "CellLogits",
    "ExpertOutput",
    # MCU
    "MCU",
    "SolveResult",
    "CombinedResult",
    # Size Resolution
    "SizeResolver",
    "resolve_size_disagreement",
    # Utils
    "create_empty_grid",
    "is_complete_grid",
    "is_partial_grid",
    "grid_to_string",
    "grid_to_hash",
    "apply_augmentation",
    # Main
    "ARCProblem",
    "solve_task",
    "evaluate_solution",
]
