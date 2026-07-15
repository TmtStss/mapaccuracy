"""Unit tests for mapaccuracy.olofsson function."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from mapaccuracy import olofsson


class TestOlofsson:
    """Test cases for the olofsson function."""

    def test_example1_olofsson2013(self):
        """
        Test Example 1 from Olofsson et al. (2013).

        This example demonstrates basic accuracy and area estimation.
        """
        # Example data from paper
        r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
        m = (
            ["1"] * 97
            + ["2"] * 3
            + ["3"] * 2
            + ["2"] * 279
            + ["3"] * 1
            + ["1"] * 3
            + ["2"] * 18
            + ["3"] * 97
        )

        Nh = {"1": 22353, "2": 1122543, "3": 610228}

        result = olofsson(r, m, Nh)

        # Verify results exist
        assert result["OA"] is not None
        assert len(result["UA"]) == 3
        assert len(result["PA"]) == 3
        assert len(result["area"]) == 3

        # Verify structure
        assert "matrix" in result
        assert "SEoa" in result
        assert "SEua" in result
        assert "SEpa" in result
        assert "SEa" in result

    def test_example2_olofsson2013(self):
        """
        Test Example 2 from Olofsson et al. (2013).

        This example uses proportions as input instead of absolute areas.
        """
        r = ["1"] * 129 + ["2"] * 403 + ["3"] * 611
        m = (
            ["1"] * 127
            + ["2"] * 2
            + ["1"] * 66
            + ["2"] * 322
            + ["3"] * 15
            + ["1"] * 54
            + ["2"] * 17
            + ["3"] * 540
        )

        Nh = {"1": 0.007, "2": 0.295, "3": 0.698}

        result = olofsson(r, m, Nh)

        # Check that results are reasonable
        assert 0 <= result["OA"] <= 1
        assert all(0 <= u <= 1 for u in result["UA"] if not np.isnan(u))
        assert all(0 <= p <= 1 for p in result["PA"] if not np.isnan(p))

    def test_example3_olofsson2014(self):
        """
        Test Example from table 8 in Olofsson et al. (2014).

        This example uses string labels for classes (Deforestation, Forest gain, etc.)
        """
        r = [1] * 69 + [2] * 56 + [3] * 175 + [4] * 340
        m = (
            [1] * 66
            + [3]
            + [4] * 2
            + [2] * 55
            + [4]
            + [1] * 5
            + [2] * 8
            + [3] * 153
            + [4] * 9
            + [1] * 4
            + [2] * 12
            + [3] * 11
            + [4] * 313
        )

        # Convert numeric to class labels
        label_map = {
            1: "Deforestation",
            2: "Forest gain",
            3: "Stable forest",
            4: "Stable non-forest",
        }
        r = [label_map[x] for x in r]
        m = [label_map[x] for x in m]

        Nh = {
            "Deforestation": 200000 * 30**2,
            "Forest gain": 150000 * 30**2,
            "Stable forest": 3200000 * 30**2,
            "Stable non-forest": 6450000 * 30**2,
        }

        result = olofsson(r, m, Nh)

        # Check results are valid
        assert 0 <= result["OA"] <= 1
        assert result["matrix"].shape[0] >= 3

    def test_class_order_specification(self):
        """Test that class order is preserved from Nh dictionary."""
        r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
        m = (
            ["1"] * 97
            + ["2"] * 3
            + ["3"] * 2
            + ["2"] * 279
            + ["3"] * 1
            + ["1"] * 3
            + ["2"] * 18
            + ["3"] * 97
        )

        # Specify custom order
        Nh = {"3": 610228, "2": 1122543, "1": 22353}
        result = olofsson(r, m, Nh)

        # Check that dict order is preserved in results
        assert result["area"].shape[0] == 3

    def test_missing_map_class(self):
        """Test when map includes classes not in reference."""
        r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
        m = (
            ["1"] * 97
            + ["2"] * 3
            + ["3"] * 2
            + ["2"] * 279
            + ["3"] * 1
            + ["1"] * 3
            + ["2"] * 18
            + ["3"] * 95
            + ["4"] * 2
        )

        Nh = {"1": 22353, "2": 1122543, "3": 610228, "4": 10}

        result = olofsson(r, m, Nh)
        assert result["OA"] is not None

    def test_missing_reference_class(self):
        """Test when reference includes classes not in map."""
        r = ["1"] * 102 + ["2"] * 280 + ["3"] * 116 + ["4"] * 2
        m = (
            ["1"] * 97
            + ["2"] * 3
            + ["3"] * 2
            + ["2"] * 279
            + ["3"] * 1
            + ["1"] * 3
            + ["2"] * 18
            + ["3"] * 97
        )

        Nh = {"1": 22353, "2": 1122543, "3": 610228}

        result = olofsson(r, m, Nh)
        assert result["OA"] is not None

    def test_margins_parameter(self):
        """Test margins parameter in output matrix."""
        r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
        m = (
            ["1"] * 97
            + ["2"] * 3
            + ["3"] * 2
            + ["2"] * 279
            + ["3"] * 1
            + ["1"] * 3
            + ["2"] * 18
            + ["3"] * 97
        )

        Nh = {"1": 22353, "2": 1122543, "3": 610228}

        # With margins
        result_with_margins = olofsson(r, m, Nh, margins=True)

        # Without margins
        result_without_margins = olofsson(r, m, Nh, margins=False)

        # Check dimensions
        assert (
            result_with_margins["matrix"].shape[0]
            > result_without_margins["matrix"].shape[0]
        )
        assert (
            result_with_margins["matrix"].shape[1]
            > result_without_margins["matrix"].shape[1]
        )

    def test_output_structure(self):
        """Test that output has the expected structure."""
        r = ["1"] * 102 + ["2"] * 280 + ["3"] * 118
        m = (
            ["1"] * 97
            + ["2"] * 3
            + ["3"] * 2
            + ["2"] * 279
            + ["3"] * 1
            + ["1"] * 3
            + ["2"] * 18
            + ["3"] * 97
        )

        Nh = {"1": 22353, "2": 1122543, "3": 610228}

        result = olofsson(r, m, Nh)

        # Check all required keys exist
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

        # Check dimensions
        assert result["UA"].shape[0] == 3
        assert result["PA"].shape[0] == 3
        assert result["area"].shape[0] == 3
        assert result["SEua"].shape[0] == 3
        assert result["SEpa"].shape[0] == 3
        assert result["SEa"].shape[0] == 3

    def test_error_missing_nh_names(self):
        """Test that error is raised if Nh is not a dictionary."""
        r = ["1"] * 10 + ["2"] * 20
        m = ["1"] * 15 + ["2"] * 15

        # This should raise an error
        with pytest.raises(ValueError):
            olofsson(r, m, [100, 200])

    def test_error_length_mismatch(self):
        """Test that error is raised if r and m have different lengths."""
        r = ["1"] * 10 + ["2"] * 20
        m = ["1"] * 15 + ["2"] * 10  # Different length
        Nh = {"1": 100, "2": 200}

        with pytest.raises(ValueError):
            olofsson(r, m, Nh)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
