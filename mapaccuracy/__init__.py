"""
mapaccuracy: Unbiased Estimators of Thematic Map Accuracy and Area

A Python package providing unbiased estimators of overall and per-class thematic map
accuracy and area for diverse sampling designs (simple, systematic and stratified).

The implemented estimators are described in:
- Olofsson et al. (2013, 2014)
- Stehman (2014)
"""

__version__ = "0.1.2"
__author__ = "Hugo Costa"

from .olofsson import olofsson
from .stehman2014 import stehman2014

__all__ = ["olofsson", "stehman2014"]
