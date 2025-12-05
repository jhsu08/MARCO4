"""
Configuration Module for MARCO4

Contains all hyperparameters and configuration settings for the
per-cell logits architecture with Dempster-Shafer combination.
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class PruningConfig:
    """Configuration for MCU-level pruning."""
    # MCU-level pruning (after D-S combination)
    mcu_prune_threshold: float = 0.05  # Prune branches below this combined mass
    max_conflict: float = 0.60  # Prune if avg conflict exceeds this
    no_progress_rounds: int = 3  # Prune if no new cells fixed for this many rounds


@dataclass
class ConfidenceConfig:
    """Configuration for confidence thresholds."""
    solution_threshold: float = 0.30  # Accept solution if mass > this
    high_confidence: float = 0.70  # Fix cell if belief > this
    medium_confidence: float = 0.50  # Consider for partial fix
    low_confidence: float = 0.30  # Uncertain, leave empty


@dataclass
class SearchConfig:
    """Configuration for A* search."""
    max_iterations: int = 1000
    max_branches: int = 100  # Max active branches in CSS
    beam_width: int = 10  # Top branches to consider per iteration

    # MCU-driven branching
    branch_threshold: float = 0.15  # Create branches for candidates above this belief
    max_branches_per_cell: int = 3  # Max branches to create per uncertain cell


@dataclass
class HeuristicConfig:
    """Configuration for A* heuristic weights."""
    # Simple heuristic based on grid completion
    # h = fraction of unfilled cells (0 = complete, 1 = empty)
    # Future: could add conflict-based penalties
    pass


@dataclass
class AugmentationConfig:
    """Configuration for Product of Experts augmentation."""
    n_augmentations: int = 8  # D4 symmetry group has 8 elements
    augmentations: tuple = (
        'identity',
        'rot90', 'rot180', 'rot270',
        'flip_h', 'flip_v',
        'flip_diag', 'flip_antidiag',
    )


@dataclass
class ExpertConfig:
    """Configuration for expert behavior."""
    num_experts: int = 3
    temperature: float = 0.7  # LLM temperature for diversity
    top_k_colors: int = 10  # All 10 colors (0-9) are considered via logits


@dataclass
class SizeResolutionConfig:
    """Configuration for hierarchical size resolution."""
    size_vote_threshold: float = 0.6  # Confidence threshold for size voting
    allow_requery: bool = True  # Allow re-querying experts with size constraint
    max_size_candidates: int = 3  # Max different sizes to consider


@dataclass
class MARCO4Config:
    """Complete configuration for MARCO4."""
    pruning: PruningConfig = field(default_factory=PruningConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    size_resolution: SizeResolutionConfig = field(default_factory=SizeResolutionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'pruning': {
                'mcu_prune_threshold': self.pruning.mcu_prune_threshold,
                'max_conflict': self.pruning.max_conflict,
                'no_progress_rounds': self.pruning.no_progress_rounds,
            },
            'confidence': {
                'solution_threshold': self.confidence.solution_threshold,
                'high_confidence': self.confidence.high_confidence,
                'medium_confidence': self.confidence.medium_confidence,
                'low_confidence': self.confidence.low_confidence,
            },
            'search': {
                'max_iterations': self.search.max_iterations,
                'max_branches': self.search.max_branches,
                'beam_width': self.search.beam_width,
                'branch_threshold': self.search.branch_threshold,
                'max_branches_per_cell': self.search.max_branches_per_cell,
            },
            'augmentation': {
                'n_augmentations': self.augmentation.n_augmentations,
            },
            'expert': {
                'num_experts': self.expert.num_experts,
                'temperature': self.expert.temperature,
                'top_k_colors': self.expert.top_k_colors,
            },
            'size_resolution': {
                'size_vote_threshold': self.size_resolution.size_vote_threshold,
                'allow_requery': self.size_resolution.allow_requery,
                'max_size_candidates': self.size_resolution.max_size_candidates,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MARCO4Config':
        """Create config from dictionary."""
        config = cls()

        if 'pruning' in config_dict:
            for k, v in config_dict['pruning'].items():
                if hasattr(config.pruning, k):
                    setattr(config.pruning, k, v)

        if 'confidence' in config_dict:
            for k, v in config_dict['confidence'].items():
                if hasattr(config.confidence, k):
                    setattr(config.confidence, k, v)

        if 'search' in config_dict:
            for k, v in config_dict['search'].items():
                if hasattr(config.search, k):
                    setattr(config.search, k, v)

        if 'heuristic' in config_dict:
            for k, v in config_dict['heuristic'].items():
                if hasattr(config.heuristic, k):
                    setattr(config.heuristic, k, v)

        if 'expert' in config_dict:
            for k, v in config_dict['expert'].items():
                if hasattr(config.expert, k):
                    setattr(config.expert, k, v)

        if 'size_resolution' in config_dict:
            for k, v in config_dict['size_resolution'].items():
                if hasattr(config.size_resolution, k):
                    setattr(config.size_resolution, k, v)

        return config


# Default configuration instance
DEFAULT_CONFIG = MARCO4Config()

# Flat dictionary for backward compatibility
DEFAULT_CONFIG_DICT = {
    # Pruning
    'mcu_prune_threshold': 0.05,
    'max_conflict': 0.60,
    'no_progress_rounds': 3,

    # Confidence
    'solution_threshold': 0.30,
    'high_confidence': 0.70,

    # Search
    'max_iterations': 1000,
    'branch_threshold': 0.15,
    'max_branches_per_cell': 3,

    # Augmentation
    'n_augmentations': 8,
}
