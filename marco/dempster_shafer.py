"""
Dempster-Shafer Theory Module for MARCO4

This module implements the core Dempster-Shafer belief functions for combining
evidence from multiple experts in the hierarchical architecture.

Key concepts:
- Mass function: m(A) assigns belief directly to subset A
- Belief function: Bel(A) = sum of m(B) for all B ⊆ A
- Plausibility: Pl(A) = sum of m(B) for all B ∩ A ≠ ∅
- Dempster's rule: Combines independent evidence sources
"""

from typing import Dict, Set, FrozenSet, Optional, List, Tuple
import math
from functools import reduce


# Type aliases for clarity
MassFunction = Dict[FrozenSet[int], float]

# Frame of discernment for ARC (colors 0-9)
THETA = frozenset(range(10))


def validate_mass_function(mass_func: MassFunction, tolerance: float = 1e-6) -> bool:
    """
    Validate that a mass function sums to 1.0 and has no negative values.

    Args:
        mass_func: Mass function to validate
        tolerance: Acceptable deviation from 1.0

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not mass_func:
        raise ValueError("Mass function cannot be empty")

    total = sum(mass_func.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Mass function sums to {total}, expected 1.0 (tolerance={tolerance})")

    for subset, mass in mass_func.items():
        if mass < 0:
            raise ValueError(f"Negative mass {mass} for subset {subset}")
        if not isinstance(subset, frozenset):
            raise ValueError(f"Keys must be frozensets, got {type(subset)}")
        if len(subset) == 0:
            raise ValueError("Empty set (∅) cannot have mass in D-S theory")

    return True


def normalize_mass_function(mass_func: MassFunction) -> MassFunction:
    """
    Normalize a mass function to sum to 1.0.

    Args:
        mass_func: Mass function to normalize

    Returns:
        Normalized mass function
    """
    total = sum(mass_func.values())
    if total == 0:
        # Return uniform over THETA if all masses are zero
        return {THETA: 1.0}
    return {k: v / total for k, v in mass_func.items()}


def dempster_combine(m1: MassFunction, m2: MassFunction) -> MassFunction:
    """
    Combine two mass functions using Dempster's rule of combination.

    Dempster's rule combines independent sources of evidence by computing
    the intersection of focal elements and normalizing by the conflict.

    Args:
        m1, m2: Mass functions mapping frozenset -> float
                Example: {frozenset([2]): 0.6, frozenset([2,3]): 0.25, frozenset(range(10)): 0.15}

    Returns:
        Combined mass function (normalized by 1-K where K is conflict)

    Algorithm:
        1. For each pair (A, B) from m1, m2:
           - If A ∩ B ≠ ∅: accumulate mass to A ∩ B
           - If A ∩ B = ∅: accumulate to conflict K
        2. Normalize: m12(C) = m12(C) / (1 - K)
        3. Handle K ≥ 0.95: return uniform distribution over THETA

    Raises:
        ValueError: If mass functions are invalid
    """
    # Validate inputs
    validate_mass_function(m1)
    validate_mass_function(m2)

    # Initialize combined mass and conflict
    combined: Dict[FrozenSet[int], float] = {}
    conflict = 0.0

    # Combine all pairs of focal elements
    for a, mass_a in m1.items():
        for b, mass_b in m2.items():
            intersection = a & b
            product = mass_a * mass_b

            if len(intersection) == 0:
                # Empty intersection contributes to conflict
                conflict += product
            else:
                # Non-empty intersection: accumulate mass
                if intersection in combined:
                    combined[intersection] += product
                else:
                    combined[intersection] = product

    # Handle high conflict: return uniform distribution
    if conflict >= 0.95:
        return {THETA: 1.0}

    # Normalize by (1 - K)
    normalization_factor = 1.0 - conflict
    if normalization_factor <= 0:
        return {THETA: 1.0}

    result = {k: v / normalization_factor for k, v in combined.items()}

    # Clean up near-zero masses
    result = {k: v for k, v in result.items() if v > 1e-10}

    # Ensure we return something valid
    if not result:
        return {THETA: 1.0}

    return result


def dempster_combine_multiple(mass_functions: List[MassFunction]) -> MassFunction:
    """
    Combine multiple mass functions using Dempster's rule.

    Args:
        mass_functions: List of mass functions to combine

    Returns:
        Combined mass function

    Raises:
        ValueError: If list is empty
    """
    if not mass_functions:
        raise ValueError("Cannot combine empty list of mass functions")

    if len(mass_functions) == 1:
        return mass_functions[0].copy()

    return reduce(dempster_combine, mass_functions)


def compute_belief(element: int, mass_func: MassFunction) -> float:
    """
    Compute the belief in a specific element (singleton).

    Belief in {x} is the sum of masses of all subsets that are subsets of {x}.
    For a singleton, this is just m({x}) since {x} is the smallest non-empty set
    containing x.

    Bel({element}) = sum of m(A) where A ⊆ {element}

    For singletons: Bel({x}) = m({x})

    Args:
        element: The element (color) to compute belief for
        mass_func: The mass function

    Returns:
        Belief value in [0, 1]
    """
    singleton = frozenset([element])
    # For singleton {x}, only m({x}) counts since no non-empty proper subset exists
    return mass_func.get(singleton, 0.0)


def compute_belief_set(subset: FrozenSet[int], mass_func: MassFunction) -> float:
    """
    Compute belief for a set (not just singleton).

    Bel(A) = sum of m(B) for all B ⊆ A

    Args:
        subset: The set to compute belief for
        mass_func: The mass function

    Returns:
        Belief value in [0, 1]
    """
    belief = 0.0
    for focal_set, mass in mass_func.items():
        if focal_set <= subset:  # focal_set is subset of subset
            belief += mass
    return belief


def compute_plausibility(element: int, mass_func: MassFunction) -> float:
    """
    Compute the plausibility of a specific element.

    Plausibility is the sum of masses of all sets that contain the element.
    Pl({x}) = sum of m(A) where x ∈ A

    Args:
        element: The element (color) to compute plausibility for
        mass_func: The mass function

    Returns:
        Plausibility value in [0, 1]
    """
    plausibility = 0.0
    for focal_set, mass in mass_func.items():
        if element in focal_set:
            plausibility += mass
    return plausibility


def compute_plausibility_set(subset: FrozenSet[int], mass_func: MassFunction) -> float:
    """
    Compute plausibility for a set.

    Pl(A) = sum of m(B) for all B ∩ A ≠ ∅

    Args:
        subset: The set to compute plausibility for
        mass_func: The mass function

    Returns:
        Plausibility value in [0, 1]
    """
    plausibility = 0.0
    for focal_set, mass in mass_func.items():
        if len(focal_set & subset) > 0:
            plausibility += mass
    return plausibility


def compute_pignistic_probability(element: int, mass_func: MassFunction) -> float:
    """
    Compute the pignistic (betting) probability for an element.

    BetP(x) = sum over A containing x of m(A) / |A|

    This transforms a belief function into a probability distribution,
    useful for decision making.

    Args:
        element: The element to compute pignistic probability for
        mass_func: The mass function

    Returns:
        Pignistic probability in [0, 1]
    """
    bet_prob = 0.0
    for focal_set, mass in mass_func.items():
        if element in focal_set:
            bet_prob += mass / len(focal_set)
    return bet_prob


def get_pignistic_distribution(mass_func: MassFunction) -> Dict[int, float]:
    """
    Get the full pignistic probability distribution over all elements.

    Args:
        mass_func: The mass function

    Returns:
        Dictionary mapping each element to its pignistic probability
    """
    # Collect all elements mentioned in the mass function
    all_elements = set()
    for focal_set in mass_func.keys():
        all_elements.update(focal_set)

    return {elem: compute_pignistic_probability(elem, mass_func)
            for elem in all_elements}


def token_probs_to_belief_mass(
    color_probs: Dict[int, float],
    strategy: str = 'entropy'
) -> MassFunction:
    """
    Convert LLM token probabilities to Dempster-Shafer mass function.

    This is a critical function that transforms neural network outputs
    (softmax probabilities) into proper D-S mass functions that can be
    combined using Dempster's rule.

    Args:
        color_probs: Dict mapping color (0-9) to probability
                    Example: {0: 0.1, 1: 0.6, 2: 0.2, 3: 0.05, 4: 0.05}
        strategy: Conversion strategy
                 - 'entropy': Use entropy to determine uncertainty
                 - 'threshold': Use fixed probability thresholds
                 - 'top_k': Use top-k probabilities

    Returns:
        Mass function: {frozenset([colors]): mass, ...}

    Strategies:
    - High confidence (max_prob > 0.6): m({best_color}) = 0.95 * max_prob
    - Medium (0.3 < max_prob ≤ 0.6): m({top_2}) with set mass
    - Low (max_prob ≤ 0.3): m({top_k}) or m(Θ) for uncertainty
    """
    if not color_probs:
        return {THETA: 1.0}

    # Normalize probabilities
    total = sum(color_probs.values())
    if total == 0:
        return {THETA: 1.0}

    probs = {k: v / total for k, v in color_probs.items()}

    # Sort by probability (descending)
    sorted_colors = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    if strategy == 'entropy':
        return _entropy_strategy(sorted_colors, probs)
    elif strategy == 'threshold':
        return _threshold_strategy(sorted_colors, probs)
    elif strategy == 'top_k':
        return _top_k_strategy(sorted_colors, probs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _entropy_strategy(
    sorted_colors: List[Tuple[int, float]],
    probs: Dict[int, float]
) -> MassFunction:
    """
    Convert probabilities using entropy to gauge uncertainty.
    """
    # Compute entropy
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)

    # Maximum entropy for uniform distribution over all colors
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    max_prob = sorted_colors[0][1]
    best_color = sorted_colors[0][0]

    mass_func: MassFunction = {}

    if normalized_entropy < 0.3:  # Low entropy = high confidence
        # High confidence in best color
        singleton_mass = 0.95 * max_prob
        mass_func[frozenset([best_color])] = singleton_mass
        mass_func[THETA] = 1.0 - singleton_mass

    elif normalized_entropy < 0.6:  # Medium entropy
        # Assign mass to top 2-3 colors
        top_colors = [c for c, p in sorted_colors[:3] if p > 0.1]
        if len(top_colors) >= 2:
            # Mass to individual top colors
            for color, prob in sorted_colors[:2]:
                mass_func[frozenset([color])] = prob * 0.6
            # Mass to the set of top colors
            mass_func[frozenset(top_colors)] = 0.2
            # Remaining to Theta
            remaining = 1.0 - sum(mass_func.values())
            if remaining > 0:
                mass_func[THETA] = remaining
        else:
            mass_func[frozenset([best_color])] = max_prob * 0.7
            mass_func[THETA] = 1.0 - max_prob * 0.7

    else:  # High entropy = high uncertainty
        # Distribute mass among top colors and Theta
        top_colors = [c for c, p in sorted_colors if p > 0.05][:5]
        if top_colors:
            mass_func[frozenset(top_colors)] = 0.4
            mass_func[THETA] = 0.6
        else:
            mass_func[THETA] = 1.0

    return normalize_mass_function(mass_func)


def _threshold_strategy(
    sorted_colors: List[Tuple[int, float]],
    probs: Dict[int, float]
) -> MassFunction:
    """
    Convert probabilities using fixed thresholds.
    """
    max_prob = sorted_colors[0][1]
    best_color = sorted_colors[0][0]

    mass_func: MassFunction = {}

    if max_prob > 0.6:  # High confidence
        mass_func[frozenset([best_color])] = 0.95 * max_prob
        mass_func[THETA] = 1.0 - 0.95 * max_prob

    elif max_prob > 0.3:  # Medium confidence
        # Top 2 colors
        top_2 = [c for c, _ in sorted_colors[:2]]
        mass_func[frozenset([best_color])] = max_prob * 0.6
        if len(sorted_colors) > 1:
            second_color, second_prob = sorted_colors[1]
            mass_func[frozenset([second_color])] = second_prob * 0.4
        mass_func[frozenset(top_2)] = 0.15
        remaining = 1.0 - sum(mass_func.values())
        if remaining > 0:
            mass_func[THETA] = remaining

    else:  # Low confidence
        # High uncertainty
        top_k = [c for c, p in sorted_colors if p > 0.05][:5]
        if top_k:
            mass_func[frozenset(top_k)] = 0.5
        mass_func[THETA] = 1.0 - sum(mass_func.values())

    return normalize_mass_function(mass_func)


def _top_k_strategy(
    sorted_colors: List[Tuple[int, float]],
    probs: Dict[int, float],
    k: int = 3
) -> MassFunction:
    """
    Convert probabilities using top-k approach.
    """
    mass_func: MassFunction = {}

    # Get top k colors with significant probability
    top_k_colors = [(c, p) for c, p in sorted_colors[:k] if p > 0.05]

    if not top_k_colors:
        return {THETA: 1.0}

    # Assign mass proportional to probability
    total_top_k_prob = sum(p for _, p in top_k_colors)

    for color, prob in top_k_colors:
        # Scale probability to leave room for uncertainty
        scaled_mass = (prob / total_top_k_prob) * 0.7
        mass_func[frozenset([color])] = scaled_mass

    # Add set mass for the top-k set
    top_k_set = frozenset(c for c, _ in top_k_colors)
    mass_func[top_k_set] = 0.15

    # Remaining to Theta
    mass_func[THETA] = 1.0 - sum(mass_func.values())

    return normalize_mass_function(mass_func)


def get_best_color(mass_func: MassFunction) -> Tuple[int, float]:
    """
    Get the color with highest belief from a mass function.

    Args:
        mass_func: The mass function

    Returns:
        Tuple of (best_color, belief)
    """
    # Collect all mentioned colors
    all_colors = set()
    for focal_set in mass_func.keys():
        all_colors.update(focal_set)

    if not all_colors:
        return (0, 0.0)

    # Use pignistic probability for decision
    best_color = None
    best_prob = -1.0

    for color in all_colors:
        prob = compute_pignistic_probability(color, mass_func)
        if prob > best_prob:
            best_prob = prob
            best_color = color

    return (best_color, best_prob)


def get_conflict_level(m1: MassFunction, m2: MassFunction) -> float:
    """
    Compute the conflict (K) between two mass functions without combining.

    Args:
        m1, m2: Mass functions

    Returns:
        Conflict level K in [0, 1]
    """
    conflict = 0.0
    for a, mass_a in m1.items():
        for b, mass_b in m2.items():
            if len(a & b) == 0:
                conflict += mass_a * mass_b
    return conflict


def mass_to_string(mass_func: MassFunction) -> str:
    """
    Convert mass function to readable string representation.

    Args:
        mass_func: The mass function

    Returns:
        Human-readable string
    """
    parts = []
    for focal_set, mass in sorted(mass_func.items(), key=lambda x: -x[1]):
        if focal_set == THETA:
            set_str = "Θ"
        elif len(focal_set) == 1:
            set_str = str(list(focal_set)[0])
        else:
            set_str = "{" + ",".join(str(c) for c in sorted(focal_set)) + "}"
        parts.append(f"m({set_str})={mass:.4f}")
    return ", ".join(parts)
