"""Unit tests for mapaccuracy.stehman2014 function."""

import numpy as np
import pandas as pd
import pytest

from mapaccuracy import stehman2014


class TestStehman2014:
    """Test cases for the stehman2014 function."""

    def test_stehman2014_example_numerical(self):
        """
        Test numerical example from Stehman (2014) paper.

        This is the main example used in the paper.
        """
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        result = stehman2014(s, r, m, Nh_strata)

        # Verify basic structure
        assert "OA" in result
        assert "UA" in result
        assert "PA" in result
        assert "area" in result
        assert result["OA"] is not None
        assert len(result["UA"]) > 0
        assert len(result["PA"]) > 0
        assert len(result["area"]) > 0

    def test_output_structure(self):
        """Test that output has expected structure."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        result = stehman2014(s, r, m, Nh_strata)

        # Check all required keys
        required_keys = [
            "OA",
            "UA",
            "PA",
            "area",
            "SEoa",
            "SEua",
            "SEpa",
            "SEa",
            "matrix",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_different_strata_and_map_classes(self):
        """Test when strata names differ from map/reference class names."""
        s = ["a"] * 5 + ["aa"] * 5 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"a": 20000, "aa": 20000, "b": 30000, "c": 20000, "d": 10000}

        result = stehman2014(s, r, m, Nh_strata)

        assert result["OA"] is not None
        assert len(result["area"]) > 0

    def test_missing_map_class(self):
        """Test when map includes classes not in reference."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 9
            + ["XX"]
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        result = stehman2014(s, r, m, Nh_strata)
        assert result["OA"] is not None

    def test_missing_reference_class(self):
        """Test when reference includes classes not in map."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["YY"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        result = stehman2014(s, r, m, Nh_strata)
        assert result["OA"] is not None

    def test_class_order_parameter(self):
        """Test specification of class order via order parameter."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        # Reverse order
        order = ["D", "C", "B", "A"]
        result = stehman2014(s, r, m, Nh_strata, order=order)

        # Check that it runs without error
        assert result["OA"] is not None

    def test_margins_parameter(self):
        """Test margins parameter in output matrix."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        # With margins
        result_with = stehman2014(s, r, m, Nh_strata, margins=True)

        # Without margins
        result_without = stehman2014(s, r, m, Nh_strata, margins=False)

        # Check dimensions
        assert result_with["matrix"].shape[0] > result_without["matrix"].shape[0]
        assert result_with["matrix"].shape[1] > result_without["matrix"].shape[1]

    def test_accuracy_bounds(self):
        """Test that accuracy estimates are between 0 and 1."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        result = stehman2014(s, r, m, Nh_strata)

        # Check overall accuracy
        assert 0 <= result["OA"] <= 1

        # Check user's and producer's accuracy (excluding NaN)
        assert all(0 <= u <= 1 for u in result["UA"] if not np.isnan(u))
        assert all(0 <= p <= 1 for p in result["PA"] if not np.isnan(p))

        # Check areas are between 0 and 1 (excluding NaN)
        assert all(0 <= a <= 1 for a in result["area"] if not np.isnan(a))

    def test_error_invalid_order(self):
        """Test that error is raised if order parameter misses required classes."""
        s = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
        m = (
            ["A"] * 7
            + ["B"] * 3
            + ["A"]
            + ["B"] * 11
            + ["C"] * 6
            + ["B"]
            + ["B"]
            + ["D"] * 10
        )
        r = (
            ["A"] * 5
            + ["C"]
            + ["B"]
            + ["A"]
            + ["B"]
            + ["C"]
            + ["A"]
            + ["B"] * 5
            + ["A"]
            + ["A"]
            + ["B"]
            + ["B"]
            + ["C"] * 5
            + ["D"]
            + ["D"]
            + ["B"]
            + ["B"]
            + ["A"]
            + ["D"] * 7
            + ["C"]
            + ["C"]
            + ["B"]
        )

        Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

        # Missing class "D" in order
        with pytest.raises(ValueError):
            stehman2014(s, r, m, Nh_strata, order=["A", "B", "C"])

    def test_error_missing_nh_strata_names(self):
        """Test that error is raised if Nh_strata is not a dictionary."""
        s = ["A"] * 10 + ["B"] * 10
        m = ["A"] * 10 + ["B"] * 10
        r = ["A"] * 10 + ["B"] * 10

        with pytest.raises(ValueError):
            stehman2014(s, r, m, [40000, 30000])

    def test_error_length_mismatch(self):
        """Test that error is raised if s, r, m have different lengths."""
        s = ["A"] * 10 + ["B"] * 10
        m = ["A"] * 10 + ["B"] * 10
        r = ["A"] * 12 + ["B"] * 9  # Different length (21 vs 20)

        Nh_strata = {"A": 40000, "B": 30000}

        with pytest.raises(ValueError):
            stehman2014(s, r, m, Nh_strata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
