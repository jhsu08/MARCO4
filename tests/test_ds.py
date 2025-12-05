"""
Unit Tests for Dempster-Shafer Theory Module

Tests cover:
- dempster_combine() with hand-calculated examples
- compute_belief() and compute_plausibility()
- token_probs_to_mass() with various distributions
- Mass function validation
"""

import pytest
import math
from marco.dempster_shafer import (
    dempster_combine,
    dempster_combine_multiple,
    compute_belief,
    compute_belief_set,
    compute_plausibility,
    compute_plausibility_set,
    compute_pignistic_probability,
    get_pignistic_distribution,
    token_probs_to_belief_mass,
    get_best_color,
    get_conflict_level,
    validate_mass_function,
    normalize_mass_function,
    mass_to_string,
    THETA,
    MassFunction,
)


class TestMassFunctionValidation:
    """Tests for mass function validation."""

    def test_valid_mass_function(self):
        """Valid mass function should pass validation."""
        m = {
            frozenset([1]): 0.6,
            frozenset([1, 2]): 0.3,
            THETA: 0.1
        }
        assert validate_mass_function(m) is True

    def test_empty_mass_function_raises(self):
        """Empty mass function should raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_mass_function({})

    def test_mass_not_summing_to_one_raises(self):
        """Mass function not summing to 1 should raise error."""
        m = {frozenset([1]): 0.5, frozenset([2]): 0.3}
        with pytest.raises(ValueError, match="sums to"):
            validate_mass_function(m)

    def test_negative_mass_raises(self):
        """Negative mass should raise error."""
        m = {frozenset([1]): -0.1, frozenset([2]): 1.1}
        with pytest.raises(ValueError, match="Negative mass"):
            validate_mass_function(m)

    def test_non_frozenset_key_raises(self):
        """Non-frozenset key should raise error."""
        m = {(1, 2): 1.0}  # tuple instead of frozenset
        with pytest.raises(ValueError, match="frozensets"):
            validate_mass_function(m)

    def test_empty_set_key_raises(self):
        """Empty set key should raise error."""
        m = {frozenset(): 0.5, frozenset([1]): 0.5}
        with pytest.raises(ValueError, match="Empty set"):
            validate_mass_function(m)


class TestNormalization:
    """Tests for mass function normalization."""

    def test_normalize_already_normalized(self):
        """Normalized mass function stays the same."""
        m = {frozenset([1]): 0.6, frozenset([2]): 0.4}
        normalized = normalize_mass_function(m)
        assert abs(normalized[frozenset([1])] - 0.6) < 1e-6
        assert abs(normalized[frozenset([2])] - 0.4) < 1e-6

    def test_normalize_unnormalized(self):
        """Unnormalized mass function gets normalized."""
        m = {frozenset([1]): 3.0, frozenset([2]): 2.0}
        normalized = normalize_mass_function(m)
        assert abs(normalized[frozenset([1])] - 0.6) < 1e-6
        assert abs(normalized[frozenset([2])] - 0.4) < 1e-6

    def test_normalize_zero_total_returns_theta(self):
        """Zero total mass returns uniform over Theta."""
        m = {frozenset([1]): 0.0, frozenset([2]): 0.0}
        normalized = normalize_mass_function(m)
        assert THETA in normalized
        assert normalized[THETA] == 1.0


class TestDempsterCombine:
    """Tests for Dempster's rule of combination."""

    def test_combine_simple_singletons(self):
        """
        Hand-calculated example with simple singletons.

        m1: m({a}) = 0.6, m(Θ) = 0.4
        m2: m({a}) = 0.8, m(Θ) = 0.2

        Combinations:
        {a} ∩ {a} = {a}: 0.6 * 0.8 = 0.48
        {a} ∩ Θ = {a}: 0.6 * 0.2 = 0.12
        Θ ∩ {a} = {a}: 0.4 * 0.8 = 0.32
        Θ ∩ Θ = Θ: 0.4 * 0.2 = 0.08

        K = 0 (no conflict)
        m12({a}) = 0.48 + 0.12 + 0.32 = 0.92
        m12(Θ) = 0.08
        """
        m1 = {frozenset([0]): 0.6, THETA: 0.4}
        m2 = {frozenset([0]): 0.8, THETA: 0.2}

        result = dempster_combine(m1, m2)

        assert abs(result[frozenset([0])] - 0.92) < 1e-6
        assert abs(result[THETA] - 0.08) < 1e-6

    def test_combine_with_conflict(self):
        """
        Example with conflict.

        m1: m({a}) = 0.8, m(Θ) = 0.2
        m2: m({b}) = 0.6, m(Θ) = 0.4

        Combinations:
        {a} ∩ {b} = ∅: 0.8 * 0.6 = 0.48 → conflict
        {a} ∩ Θ = {a}: 0.8 * 0.4 = 0.32
        Θ ∩ {b} = {b}: 0.2 * 0.6 = 0.12
        Θ ∩ Θ = Θ: 0.2 * 0.4 = 0.08

        K = 0.48
        Normalization: 1 - K = 0.52

        m12({a}) = 0.32 / 0.52 ≈ 0.615
        m12({b}) = 0.12 / 0.52 ≈ 0.231
        m12(Θ) = 0.08 / 0.52 ≈ 0.154
        """
        m1 = {frozenset([0]): 0.8, THETA: 0.2}
        m2 = {frozenset([1]): 0.6, THETA: 0.4}

        result = dempster_combine(m1, m2)

        expected_a = 0.32 / 0.52
        expected_b = 0.12 / 0.52
        expected_theta = 0.08 / 0.52

        assert abs(result[frozenset([0])] - expected_a) < 1e-6
        assert abs(result[frozenset([1])] - expected_b) < 1e-6
        assert abs(result[THETA] - expected_theta) < 1e-6

    def test_combine_with_sets(self):
        """
        Example with set-valued focal elements.

        m1: m({a,b}) = 0.7, m(Θ) = 0.3
        m2: m({b,c}) = 0.5, m(Θ) = 0.5

        Combinations:
        {a,b} ∩ {b,c} = {b}: 0.7 * 0.5 = 0.35
        {a,b} ∩ Θ = {a,b}: 0.7 * 0.5 = 0.35
        Θ ∩ {b,c} = {b,c}: 0.3 * 0.5 = 0.15
        Θ ∩ Θ = Θ: 0.3 * 0.5 = 0.15

        K = 0 (no empty intersections)
        """
        m1 = {frozenset([0, 1]): 0.7, THETA: 0.3}
        m2 = {frozenset([1, 2]): 0.5, THETA: 0.5}

        result = dempster_combine(m1, m2)

        assert abs(result[frozenset([1])] - 0.35) < 1e-6
        assert abs(result[frozenset([0, 1])] - 0.35) < 1e-6
        assert abs(result[frozenset([1, 2])] - 0.15) < 1e-6
        assert abs(result[THETA] - 0.15) < 1e-6

    def test_high_conflict_returns_uniform(self):
        """High conflict (K >= 0.95) should return uniform over Theta."""
        m1 = {frozenset([0]): 0.99, THETA: 0.01}
        m2 = {frozenset([1]): 0.99, THETA: 0.01}

        result = dempster_combine(m1, m2)

        # Conflict K ≈ 0.99 * 0.99 = 0.98 > 0.95
        assert THETA in result
        assert result[THETA] == 1.0

    def test_combine_multiple(self):
        """Test combining multiple mass functions."""
        m1 = {frozenset([0]): 0.5, THETA: 0.5}
        m2 = {frozenset([0]): 0.6, THETA: 0.4}
        m3 = {frozenset([0]): 0.7, THETA: 0.3}

        result = dempster_combine_multiple([m1, m2, m3])

        # Should have high belief in {0}
        assert frozenset([0]) in result
        assert result[frozenset([0])] > 0.9

    def test_combine_single_returns_copy(self):
        """Combining single mass function returns copy."""
        m = {frozenset([0]): 0.6, THETA: 0.4}
        result = dempster_combine_multiple([m])

        assert result[frozenset([0])] == 0.6
        assert result[THETA] == 0.4

    def test_combine_empty_raises(self):
        """Combining empty list raises error."""
        with pytest.raises(ValueError):
            dempster_combine_multiple([])


class TestBeliefAndPlausibility:
    """Tests for belief and plausibility functions."""

    def test_belief_singleton(self):
        """Belief in singleton equals its mass."""
        m = {
            frozenset([1]): 0.3,
            frozenset([1, 2]): 0.4,
            THETA: 0.3
        }
        # Bel({1}) = m({1}) = 0.3 (only {1} is subset of {1})
        assert abs(compute_belief(1, m) - 0.3) < 1e-6

    def test_plausibility_singleton(self):
        """Plausibility sums all sets containing element."""
        m = {
            frozenset([1]): 0.3,
            frozenset([1, 2]): 0.4,
            THETA: 0.3
        }
        # Pl({1}) = m({1}) + m({1,2}) + m(Θ) = 0.3 + 0.4 + 0.3 = 1.0
        assert abs(compute_plausibility(1, m) - 1.0) < 1e-6

    def test_plausibility_disjoint(self):
        """Plausibility of element not in any focal element."""
        m = {
            frozenset([1]): 0.6,
            frozenset([2]): 0.4
        }
        # Pl({3}) = 0 (3 not in any focal element)
        assert compute_plausibility(3, m) == 0.0

    def test_belief_set(self):
        """Test belief for a set."""
        m = {
            frozenset([1]): 0.2,
            frozenset([2]): 0.3,
            frozenset([1, 2]): 0.2,
            THETA: 0.3
        }
        # Bel({1,2}) = m({1}) + m({2}) + m({1,2}) = 0.2 + 0.3 + 0.2 = 0.7
        subset = frozenset([1, 2])
        assert abs(compute_belief_set(subset, m) - 0.7) < 1e-6

    def test_plausibility_set(self):
        """Test plausibility for a set."""
        m = {
            frozenset([1]): 0.2,
            frozenset([2]): 0.3,
            frozenset([3]): 0.2,
            THETA: 0.3
        }
        # Pl({1,2}) = m({1}) + m({2}) + m(Θ) = 0.2 + 0.3 + 0.3 = 0.8
        subset = frozenset([1, 2])
        assert abs(compute_plausibility_set(subset, m) - 0.8) < 1e-6


class TestPignisticProbability:
    """Tests for pignistic (betting) probability."""

    def test_pignistic_singleton(self):
        """Pignistic probability for singleton focal elements."""
        m = {
            frozenset([1]): 0.6,
            frozenset([2]): 0.4
        }
        # BetP(1) = m({1})/1 = 0.6
        # BetP(2) = m({2})/1 = 0.4
        assert abs(compute_pignistic_probability(1, m) - 0.6) < 1e-6
        assert abs(compute_pignistic_probability(2, m) - 0.4) < 1e-6

    def test_pignistic_with_sets(self):
        """Pignistic probability with set-valued focal elements."""
        m = {
            frozenset([1]): 0.3,
            frozenset([1, 2]): 0.4,
            THETA: 0.3  # Θ = {0,1,...,9}
        }
        # BetP(1) = 0.3/1 + 0.4/2 + 0.3/10 = 0.3 + 0.2 + 0.03 = 0.53
        assert abs(compute_pignistic_probability(1, m) - 0.53) < 1e-6

    def test_pignistic_distribution_sums_to_one(self):
        """Pignistic distribution should sum to 1."""
        m = {
            frozenset([1]): 0.3,
            frozenset([1, 2]): 0.4,
            frozenset([2, 3]): 0.3
        }
        dist = get_pignistic_distribution(m)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6


class TestTokenProbsToMass:
    """Tests for converting token probabilities to mass functions."""

    def test_high_confidence_conversion(self):
        """High confidence probabilities create singleton mass."""
        probs = {0: 0.8, 1: 0.1, 2: 0.05, 3: 0.05}
        mass = token_probs_to_belief_mass(probs, strategy='entropy')

        # Should have high mass on singleton {0}
        assert frozenset([0]) in mass
        assert mass[frozenset([0])] > 0.5

    def test_low_confidence_conversion(self):
        """Low confidence probabilities create set mass."""
        probs = {i: 0.1 for i in range(10)}
        mass = token_probs_to_belief_mass(probs, strategy='entropy')

        # Should have significant mass on Theta
        assert THETA in mass
        assert mass[THETA] > 0.3

    def test_medium_confidence_conversion(self):
        """Medium confidence creates mix of singleton and set mass."""
        probs = {0: 0.4, 1: 0.35, 2: 0.25}
        mass = token_probs_to_belief_mass(probs, strategy='threshold')

        # Should have mass on individual colors and possibly sets
        total = sum(mass.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_probs_returns_theta(self):
        """Empty probabilities return uniform over Theta."""
        mass = token_probs_to_belief_mass({})
        assert THETA in mass
        assert mass[THETA] == 1.0

    def test_mass_sums_to_one(self):
        """Resulting mass function should sum to 1."""
        for strategy in ['entropy', 'threshold', 'top_k']:
            probs = {0: 0.5, 1: 0.3, 2: 0.2}
            mass = token_probs_to_belief_mass(probs, strategy=strategy)
            total = sum(mass.values())
            assert abs(total - 1.0) < 1e-6, f"Strategy {strategy} failed"


class TestGetBestColor:
    """Tests for getting best color from mass function."""

    def test_singleton_best(self):
        """Best color from singleton mass."""
        m = {frozenset([3]): 0.8, THETA: 0.2}
        color, prob = get_best_color(m)
        assert color == 3
        assert prob > 0.8

    def test_tie_breaking(self):
        """Best color with tie should pick one consistently."""
        m = {frozenset([1]): 0.5, frozenset([2]): 0.5}
        color, prob = get_best_color(m)
        assert color in [1, 2]
        assert abs(prob - 0.5) < 1e-6


class TestConflictLevel:
    """Tests for computing conflict between mass functions."""

    def test_no_conflict(self):
        """No conflict when masses agree."""
        m1 = {frozenset([1]): 0.6, THETA: 0.4}
        m2 = {frozenset([1]): 0.8, THETA: 0.2}
        K = get_conflict_level(m1, m2)
        assert K == 0.0

    def test_full_conflict(self):
        """Full conflict when masses disagree on singletons."""
        m1 = {frozenset([1]): 1.0}
        m2 = {frozenset([2]): 1.0}
        K = get_conflict_level(m1, m2)
        assert K == 1.0

    def test_partial_conflict(self):
        """Partial conflict."""
        m1 = {frozenset([1]): 0.8, THETA: 0.2}
        m2 = {frozenset([2]): 0.6, THETA: 0.4}
        K = get_conflict_level(m1, m2)
        # K = 0.8 * 0.6 = 0.48
        assert abs(K - 0.48) < 1e-6


class TestMassToString:
    """Tests for mass function string representation."""

    def test_string_representation(self):
        """Test readable string output."""
        m = {frozenset([1]): 0.6, THETA: 0.4}
        s = mass_to_string(m)
        assert "0.6" in s
        assert "0.4" in s

    def test_singleton_display(self):
        """Singleton sets display as single number."""
        m = {frozenset([5]): 1.0}
        s = mass_to_string(m)
        assert "5" in s

    def test_theta_display(self):
        """Theta displays as Θ."""
        m = {THETA: 1.0}
        s = mass_to_string(m)
        assert "Θ" in s
