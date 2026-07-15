"""Olofsson et al. thematic map accuracy and area estimators."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from .internal import _check_labels, _check_length


def olofsson(r, m, Nh, margins=True):
    """
    Thematic map accuracy and area estimators (Olofsson et al. 2013, 2014).

    Implements the estimators described in Olofsson et al. (2013, 2014) for overall
    accuracy, producer's accuracy, user's accuracy, and area. Includes precision estimates.

    Argument Nh must be a dictionary with class names as keys to explicitly and clearly
    identify the class that each area refers to. The order of keys will be used for
    displaying the results.

    In the error matrix returned, entries corresponding to no observed cases will present
    NaN rather than 0. This is to emphasize the difference between the absence of cases
    and the presence of some (few) cases that represent a very small proportion of area
    (almost zero) and thus possibly rounded to zero. However, NaN means zero proportion
    of area.

    Parameters
    ----------
    r : array-like
        Reference class labels. Will be converted to list.
    m : array-like
        Map class labels. Will be converted to list.
    Nh : dict
        Area, number of pixels, or proportion of the classes in the map.
        Keys are class labels.
    margins : bool, default True
        If False, the error matrix produced includes no margins (sum of rows and columns).

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'OA': overall accuracy
        - 'UA': user's accuracy
        - 'PA': producer's accuracy
        - 'area': area proportion
        - 'SEoa': standard error of OA
        - 'SEua': standard error of UA
        - 'SEpa': standard error of PA
        - 'SEa': standard error of area proportion
        - 'matrix': confusion error (area proportion). Rows and columns represent map
                    and reference class labels, respectively

    References
    ----------
    Olofsson, P.; Foody, G. M.; Stehman, S. V.; Woodcock, C. E. (2013).
    Making better use of accuracy data in land change studies: Estimating accuracy
    and area and quantifying uncertainty using stratified estimation.
    Remote Sens. Environ., 129, 122-131.

    Olofsson, P.; Foody, G. M.; Herold, M.; Stehman, S. V.; Woodcock, C. E.;
    Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy
    of land change. Remote Sens. Environ., 148, 42-57.
    """

    # Check and convert arguments
    r = list(r) if hasattr(r, "__iter__") and not isinstance(r, str) else [r]
    m = list(m) if hasattr(m, "__iter__") and not isinstance(m, str) else [m]

    if not isinstance(Nh, dict):
        raise ValueError("Nh must be a dictionary with named keys.")

    class_names = list(Nh.keys())
    r_values = np.array(r, dtype=object)
    m_values = np.array(m, dtype=object)

    # Check labels
    _check_labels(class_names, m_values)
    _check_length(r, m)

    # Add classes from r that are not in Nh
    temp = set(r_values) - set(class_names)
    if len(temp) > 0:
        for t in temp:
            Nh[t] = 0
        class_names = list(Nh.keys())

    # Create factors with specified levels
    r_factor = pd.Categorical(r_values, categories=class_names, ordered=False)
    m_factor = pd.Categorical(m_values, categories=class_names, ordered=False)

    # Error matrix (contingency table)
    matrix = pd.crosstab(m_factor, r_factor)
    # Ensure all classes appear in both rows and columns (like R's table with factors)
    matrix = matrix.reindex(index=class_names, columns=class_names, fill_value=0)

    # Sampling settings
    q = len(class_names)  # number of classes
    # Use float to avoid integer overflow with large Nh values
    Nh = {k: float(v) for k, v in Nh.items()}
    A = sum(Nh.values())  # total area
    nh = matrix.sum(axis=1).values.astype(float)  # stratum sample size
    Wh = np.array([Nh[c] / A for c in class_names])  # map class proportions

    # Confusion matrix (estimated area proportions)
    # Calculate proportions by dividing each row by its sum
    props = matrix.div(matrix.sum(axis=1), axis=0)  # proportions by map class

    # Zero out rows/cols corresponding to reference classes not in map
    for t in temp:
        props.loc[t, t] = 0

    # Replace NaN with NA representation and apply Wh weights
    props = props.replace([np.inf, -np.inf], np.nan)
    props_array = props.values
    for j in range(q):
        props_array[j, :] = Wh[j] * props_array[j, :]

    props = pd.DataFrame(props_array, index=matrix.index, columns=matrix.columns)

    # Accuracy and area estimates
    OA = np.nansum(np.diag(props_array))

    # User's accuracy
    row_sums = np.nansum(props_array, axis=1)
    UA = np.diag(props_array) / np.where(row_sums > 0, row_sums, np.nan)

    # Producer's accuracy
    col_sums = np.nansum(props_array, axis=0)
    PA = np.diag(props_array) / np.where(col_sums > 0, col_sums, np.nan)

    tarea = col_sums

    # Standard error of OA
    VAR_OA = 0
    for j in range(q):
        if nh[j] > 1:
            temp_var = Wh[j] ** 2 * UA[j] * (1 - UA[j]) / (nh[j] - 1)
            if np.isnan(temp_var) or np.isinf(temp_var):
                temp_var = 0
            VAR_OA += temp_var

    SEoa = np.sqrt(VAR_OA)

    # Standard error of UA
    VAR_UA = []
    for j in range(q):
        if nh[j] > 1:
            temp_var = UA[j] * (1 - UA[j]) / (nh[j] - 1)
        else:
            temp_var = 0
        if np.isnan(temp_var):
            temp_var = 0
        VAR_UA.append(temp_var)

    SEua = np.sqrt(np.array(VAR_UA))

    # Standard error of PA
    N_j = []
    for j in range(q):
        temp = 0
        for i in range(q):
            if nh[i] > 0:
                temp += Nh[class_names[i]] / nh[i] * matrix.iloc[i, j]
        N_j.append(temp)

    VAR_PA = []
    for j in range(q):
        temp1 = (
            Nh[class_names[j]] ** 2
            * (1 - PA[j]) ** 2
            * UA[j]
            * (1 - UA[j])
            / (nh[j] - 1)
        )
        if np.isnan(temp1) or np.isinf(temp1):
            temp1 = 0

        temp2 = 0
        for i in range(q):
            if i != j and nh[i] > 1:
                with np.errstate(invalid="ignore"):
                    val = (
                        Nh[class_names[i]] ** 2
                        * matrix.iloc[i, j]
                        / nh[i]
                        * (1 - matrix.iloc[i, j] / nh[i])
                        / (nh[i] - 1)
                    )
                if np.isnan(val) or np.isinf(val):
                    val = 0
                temp2 += val

        if np.isnan(temp2) or np.isinf(temp2):
            temp2 = 0

        if N_j[j] == 0:
            VAR_PA.append(np.nan)
        else:
            VAR_PA.append((1 / N_j[j] ** 2) * (temp1 + PA[j] ** 2 * temp2))

    SEpa = np.sqrt(np.array(VAR_PA))

    # Standard error of the area
    VAR_A = []
    for j in range(q):
        a = Wh**2
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where(nh > 0, matrix.iloc[:, j].values / nh, 0.0)
            c = np.where(nh > 0, 1 - matrix.iloc[:, j].values / nh, 0.0)
            d = np.where(nh > 1, nh - 1, np.nan)
        var_aj = np.nansum(a * (b * c / d))
        VAR_A.append(var_aj)

    SEa = np.sqrt(np.array(VAR_A))

    # Add margins to confusion matrix if requested
    props_matrix = props.copy()
    props_matrix[matrix == 0] = np.nan  # Set to NaN where no observations

    if margins:
        # Add row and column sums
        row_sums_margins = props_matrix.sum(axis=1, skipna=True)
        col_sums_margins = props_matrix.sum(axis=0, skipna=True)

        props_matrix["sum"] = row_sums_margins
        sum_row = col_sums_margins.tolist() + [col_sums_margins.sum()]
        props_matrix.loc["sum"] = sum_row

    return {
        "OA": OA,
        "UA": UA,
        "PA": PA,
        "area": tarea,
        "SEoa": SEoa,
        "SEua": SEua,
        "SEpa": SEpa,
        "SEa": SEa,
        "matrix": props_matrix,
    }
