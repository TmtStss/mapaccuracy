# mapaccuracy

Unbiased estimators of overall and per-class thematic map accuracy and area for diverse sampling designs (simple, systematic and stratified).


**IMPORTANT DISCLAIMER.**
This repository is an unofficial, LLM-assisted Python port of the
original [`mapaccuracy`](https://github.com/hagc/mapaccuracy) R package
by Hugo Costa. The original R package is the authoritative upstream implementation
and may include features, fixes, or updates not available in this port.
The Python implementation has been reviewed and tested against the
numerical examples covered by the test suite. Small numerical
differences may occur because of implementation details and
floating-point arithmetic. This project is not affiliated with or endorsed by the original author.

## Overview

Implemented estimators are described in:

* Olofsson, P.; Foody, G. M.; Herold, M.; Stehman, S. V.; Woodcock, C. E.; Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sens. Environ.*, 148, 42–57, doi:10.1016/j.rse.2014.02.015.

* Olofsson, P.; Foody, G. M.; Stehman, S. V.; Woodcock, C. E. (2013). Making better use of accuracy data in land change studies: Estimating accuracy and area and quantifying uncertainty using stratified estimation. *Remote Sens. Environ.*, 129, 122–131, doi:10.1016/j.rse.2012.10.031.

* Stehman, S. V. (2014). Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes. *Int. J. Remote Sens.*, 35, 4923–4939, doi:10.1080/01431161.2014.930207.

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/TmtStss/mapaccuracy.git
```

Alternatively, clone the repository and install it locally:
```bash
git clone https://github.com/TmtStss/mapaccuracy.git
cd mapaccuracy
pip install .
```

## Quick Start

### Example 1: Basic accuracy assessment with the Olofsson method

```python
from mapaccuracy import olofsson

# Reference and map class labels
r = ["1"]*102 + ["2"]*280 + ["3"]*118
m = ["1"]*97 + ["2"]*3 + ["3"]*2 + ["2"]*279 + ["3"]*1 + ["1"]*3 + ["2"]*18 + ["3"]*97

# Area (or pixel count, or proportion) for each map class
Nh = {"1": 22353, "2": 1122543, "3": 610228}

# Compute estimates
result = olofsson(r, m, Nh)

# Access results
print(f"Overall Accuracy: {result['OA']:.4f}")
print(f"User's Accuracy: {result['UA']}")
print(f"Producer's Accuracy: {result['PA']}")
print(f"Estimated Areas: {result['area']}")
print(f"\nError Matrix:\n{result['matrix']}")
```

### Example 2: Stratified sampling with Stehman method

```python
from mapaccuracy import stehman2014

# Stratum, reference, and map class labels
s = ["A"]*10 + ["B"]*10 + ["C"]*10 + ["D"]*10
r = ["A"]*5 + ["C"] + ["B"] + ["A"] + ["B"] + ["C"] + \
    ["A"] + ["B"]*5 + ["A"] + ["A"] + ["B"] + \
    ["B"] + ["C"]*5 + ["D"] + ["D"] + ["B"] + ["B"] + ["A"] + \
    ["D"]*7 + ["C"] + ["C"] + ["B"]
m = ["A"]*7 + ["B"]*3 + "A" + \
    ["B"]*11 + ["C"]*6 + "B" + "B" + ["D"]*10

# Area of each stratum
Nh_strata = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}

# Compute estimates
result = stehman2014(s, r, m, Nh_strata)

print(f"Overall Accuracy: {result['OA']:.4f}")
print(f"Producer's Accuracy: {result['PA']}")
```

## Functions

### `olofsson(r, m, Nh, margins=True)`

Implements the Olofsson et al. (2013, 2014) estimators.

**Parameters:**
- `r`: Reference class labels
- `m`: Map class labels
- `Nh`: Dictionary with named class areas/counts. Keys must be class labels.
- `margins`: If True, add margin sums to result matrix

**Returns:**
Dictionary with keys:
- `OA`: Overall accuracy
- `UA`: User's accuracy per class
- `PA`: Producer's accuracy per class
- `area`: Estimated area proportion per class
- `SEoa`: Standard error of overall accuracy
- `SEua`: Standard errors of user's accuracy
- `SEpa`: Standard errors of producer's accuracy
- `SEa`: Standard errors of area estimates
- `matrix`: Confusion/error matrix as pandas DataFrame

### `stehman2014(s, r, m, Nh_strata, margins=True, order=None)`

Implements the Stehman (2014) estimator for stratified random sampling.

**Parameters:**
- `s`: Stratum class labels
- `r`: Reference class labels
- `m`: Map class labels
- `Nh_strata`: Dictionary with named stratum sizes. Keys must be stratum labels.
- `margins`: If True, add margin sums to result matrix
- `order`: Order of classes in output. If None, uses sorted union of r and m.

**Returns:**
Dictionary with same structure as `olofsson()` output

## Notes

- Input class labels can be strings or numbers
- The order of keys in `Nh` or `Nh_strata` dictionaries determines the order of results
- NA/NaN values in the error matrix indicate cells with no observations
- Standard errors can be used to construct confidence intervals

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=mapaccuracy --cov-report=html
```

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).

It is an unofficial Python port of the original
[`mapaccuracy`](https://github.com/hagc/mapaccuracy) R package, which is
also distributed under the MIT License.

The original copyright notice is retained:

Copyright (c) 2021 Hugo Costa

See [`NOTICE`](NOTICE) for attribution and project provenance.

## Citation

If you use this package in your research, please cite the original papers:

Olofsson et al. (2013, 2014) and/or Stehman (2014) depending on which estimator you use.

## See Also

- Original R package: https://github.com/hagc/mapaccuracy

## Author

Python port maintained by Timothée Stassin.