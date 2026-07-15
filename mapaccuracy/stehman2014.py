"""Stehman 2014 thematic map accuracy and area estimators for stratified random sampling."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from .internal import _check_labels, _check_length


def stehman2014(s, r, m, Nh_strata, margins=True, order=None):
    """
    Thematic map accuracy and area under stratified random sampling when strata differ
    from map classes.

    Implements the estimators described in Stehman (2014) for overall accuracy, producer's
    accuracy, user's accuracy, and area. Includes precision estimates.

    Argument Nh_strata must be a dictionary with stratum names as keys to explicitly and
    clearly identify the stratum that each entry refers to. The keys of Nh_strata are
    expected to match the strata class labels of argument s.

    In the error matrix returned, entries corresponding to no observed cases will present
    NaN rather than 0.

    Parameters
    ----------
    s : array-like
        Strata class labels. Will be converted to list.
    r : array-like
        Reference class labels. Will be converted to list.
    m : array-like
        Map class labels. Will be converted to list.
    Nh_strata : dict
        Number of pixels forming each stratum. Keys are stratum labels.
    margins : bool, default True
        If False, the error matrix produced includes no margins.
    order : list, optional
        Classes to be displayed in results and their sequence.
        If missing, equal to sorted(union(r, m)).

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
        - 'matrix': confusion error (area proportion)

    References
    ----------
    Stehman, S. V. (2014). Estimating area and map accuracy for stratified random
    sampling when the strata are different from the map classes. Int. J. Remote Sens.,
    35, 4923-4939.
    """

    # Check and convert arguments
    s = list(s) if hasattr(s, "__iter__") and not isinstance(s, str) else [s]
    r = list(r) if hasattr(r, "__iter__") and not isinstance(r, str) else [r]
    m = list(m) if hasattr(m, "__iter__") and not isinstance(m, str) else [m]

    if not isinstance(Nh_strata, dict):
        raise ValueError("Nh_strata must be a dictionary with named keys.")

    # Check labels and length
    strata_names = list(Nh_strata.keys())
    r_values = np.array(r, dtype=object)
    m_values = np.array(m, dtype=object)
    s_values = np.array(s, dtype=object)

    _check_labels(strata_names, s_values)
    _check_length(s, r, m)

    # Keep only strata found in s
    Nh_strata = {k: v for k, v in Nh_strata.items() if k in s_values}
    strata_names = list(Nh_strata.keys())

    # Set order of classes
    if order is None:
        order = sorted(set(r_values) | set(m_values))
    else:
        # Check that order includes all classes
        missing = (set(r_values) | set(m_values)) - set(order)
        if missing:
            raise ValueError(
                f"Argument 'order' must include the following label(s): "
                f"{'; '.join(map(str, sorted(missing)))}"
            )

    # Create factors
    s_factor = pd.Categorical(s_values, categories=strata_names, ordered=False)
    r_factor = pd.Categorical(r_values, categories=order, ordered=False)
    m_factor = pd.Categorical(m_values, categories=order, ordered=False)

    # Create sample dataframe
    sample = pd.DataFrame({"s": s_factor, "m": m_factor, "r": r_factor})

    # Calculate inclusion probabilities
    nh = sample["s"].value_counts().sort_index()
    nh = nh[nh.index.isin(strata_names)]  # ensure order matches Nh_strata

    if (nh < 2).any():
        warning_strata = nh[nh < 2].index.tolist()
        print(
            f"Warning: The following strata include only one observation: "
            f"{'; '.join(map(str, warning_strata))}"
        )

    Nh_array = np.array([Nh_strata[s] for s in nh.index], dtype=float)
    incl_prob = nh.values / Nh_array

    # Map inclusion probabilities to each sample
    sample["incl"] = sample["s"].map(pd.Series(incl_prob, index=nh.index)).astype(float)

    classes_m = list(m_factor.categories)
    classes_r = list(r_factor.categories)
    classes_s = list(s_factor.categories)

    # ===== CASE 1: Overall accuracy, area proportions, and error matrix cells =====

    # Overall accuracy indicator
    sample["yu_O"] = (sample["m"] == sample["r"]).astype(float)

    # Reference class indicators
    for i, c in enumerate(classes_r):
        col_name = f"yu_r{i}"
        sample[col_name] = (sample["r"] == c).astype(float)

    # Error matrix cell indicators
    for i, m_class in enumerate(classes_m):
        for j, r_class in enumerate(classes_r):
            col_name = f"yu_m{i}r{j}"
            sample[col_name] = (
                (sample["m"] == m_class) & (sample["r"] == r_class)
            ).astype(float)

    # Equation 2 of Stehman 2014: proportion calculation
    def eq2(col_name):
        return np.sum(sample[col_name].values / sample["incl"].values) / np.sum(
            Nh_array
        )

    # Overall accuracy
    O = eq2("yu_O")

    # Proportions of area for reference classes
    A = np.array([eq2(f"yu_r{i}") for i in range(len(classes_r))])

    # Error matrix (proportion of area with map class i and reference class j)
    M = np.zeros((len(classes_m), len(classes_r)))
    for i in range(len(classes_m)):
        for j in range(len(classes_r)):
            col_name = f"yu_m{i}r{j}"
            val = eq2(col_name)
            M[i, j] = np.nan if val == 0 else val

    # Standard error for overall accuracy
    group_means = sample.groupby("s", observed=False)[
        ["yu_O"]
        + [f"yu_r{i}" for i in range(len(classes_r))]
        + [f"yu_m{i}r{j}" for i in range(len(classes_m)) for j in range(len(classes_r))]
    ].mean()

    group_vars = sample.groupby("s", observed=False)[
        ["yu_O"]
        + [f"yu_r{i}" for i in range(len(classes_r))]
        + [f"yu_m{i}r{j}" for i in range(len(classes_m)) for j in range(len(classes_r))]
    ].var()

    group_vars = group_vars.fillna(
        0
    )  # NaN variance becomes 0 for strata with 1 observation

    # Equation 25 of Stehman 2014
    def eq25(col_name):
        var_col = group_vars[col_name].values
        result = np.sum(Nh_array**2 * (1 - nh.values / Nh_array) * var_col / nh.values)
        result = result * (1 / np.sum(Nh_array) ** 2)
        return result

    Vo = eq25("yu_O")
    SEo = np.sqrt(Vo)

    # Standard errors for reference class proportions
    Va = np.array([eq25(f"yu_r{i}") for i in range(len(classes_r))])
    SEa = np.sqrt(Va)

    # ===== CASE 2: User's and Producer's accuracy (ratio estimation) =====

    # User's accuracy indicators
    for i, m_class in enumerate(classes_m):
        col_name = f"yu_U{i}"
        sample[col_name] = (
            (sample["m"] == m_class) & (sample["m"] == sample["r"])
        ).astype(float)

    for i, m_class in enumerate(classes_m):
        col_name = f"xu_U{i}"
        sample[col_name] = (sample["m"] == m_class).astype(float)

    # Producer's accuracy indicators
    for j, r_class in enumerate(classes_r):
        col_name = f"yu_P{j}"
        sample[col_name] = (
            (sample["r"] == r_class) & (sample["r"] == sample["m"])
        ).astype(float)

    for j, r_class in enumerate(classes_r):
        col_name = f"xu_P{j}"
        sample[col_name] = (sample["r"] == r_class).astype(float)

    # Recalculate aggregates with new columns
    group_means = sample.groupby("s", observed=False).mean(numeric_only=True)
    group_vars = sample.groupby("s", observed=False).var(numeric_only=True)
    group_vars = group_vars.fillna(0)

    # User's accuracy
    U_numerator = np.array(
        [
            np.sum(group_means[f"yu_U{i}"].values * Nh_array)
            for i in range(len(classes_m))
        ]
    )

    U_denominator = np.array(
        [
            np.sum(group_means[f"xu_U{i}"].values * Nh_array)
            for i in range(len(classes_m))
        ]
    )

    with np.errstate(invalid="ignore"):
        U = np.where(U_denominator != 0, U_numerator / U_denominator, np.nan)

    # Producer's accuracy
    P_numerator = np.array(
        [
            np.sum(group_means[f"yu_P{j}"].values * Nh_array)
            for j in range(len(classes_r))
        ]
    )

    P_denominator = np.array(
        [
            np.sum(group_means[f"xu_P{j}"].values * Nh_array)
            for j in range(len(classes_r))
        ]
    )

    with np.errstate(invalid="ignore"):
        P = np.where(P_denominator != 0, P_numerator / P_denominator, np.nan)

    # Standard errors for user's accuracy and producer's accuracy
    # Equation 28 of Stehman 2014
    def eq28(R, X, vary, varx, cov):
        if X == 0 or np.isnan(R):
            return np.nan
        result = np.sum(
            Nh_array**2
            * (1 - nh.values / Nh_array)
            * (vary + R**2 * varx - 2 * R * cov)
            / nh.values
        )
        result = result * (1 / X**2)
        return result

    # Calculate covariances
    SE_U = []
    for i in range(len(classes_m)):
        col_yu = f"yu_U{i}"
        col_xu = f"xu_U{i}"

        # Group by stratum and calculate variance/covariance
        cov_values = []
        for strata in classes_s:
            strata_data = sample[sample["s"] == strata]
            if len(strata_data) >= 2:
                cov_val = np.cov(strata_data[col_yu], strata_data[col_xu])[0, 1]
            else:
                cov_val = 0
            cov_values.append(cov_val)

        vary = group_vars[col_yu].values
        varx = group_vars[col_xu].values

        var_u = eq28(U[i], U_denominator[i], vary, varx, np.array(cov_values))
        SE_U.append(np.sqrt(var_u) if not np.isnan(var_u) else np.nan)

    SEu = np.array(SE_U)

    # Calculate standard errors for producer's accuracy
    SE_P = []
    for j in range(len(classes_r)):
        col_yp = f"yu_P{j}"
        col_xp = f"xu_P{j}"

        cov_values = []
        for strata in classes_s:
            strata_data = sample[sample["s"] == strata]
            if len(strata_data) >= 2:
                cov_val = np.cov(strata_data[col_yp], strata_data[col_xp])[0, 1]
            else:
                cov_val = 0
            cov_values.append(cov_val)

        vary = group_vars[col_yp].values
        varx = group_vars[col_xp].values

        var_p = eq28(P[j], P_denominator[j], vary, varx, np.array(cov_values))
        SE_P.append(np.sqrt(var_p) if not np.isnan(var_p) else np.nan)

    SEp = np.array(SE_P)

    # Create error matrix dataframe
    matrix_df = pd.DataFrame(
        M,
        index=pd.Index(classes_m, name="map"),
        columns=pd.Index(classes_r, name="reference"),
    )

    # Set 0 values to NaN
    matrix_df[matrix_df == 0] = np.nan

    # Add margins if requested
    if margins:
        row_sums = matrix_df.sum(axis=1, skipna=True)
        col_sums = matrix_df.sum(axis=0, skipna=True)

        matrix_df["sum"] = row_sums
        sum_row = col_sums.tolist() + [col_sums.sum()]
        matrix_df.loc["sum"] = sum_row

    return {
        "OA": O,
        "UA": U,
        "PA": P,
        "area": A,
        "SEoa": SEo,
        "SEua": SEu,
        "SEpa": SEp,
        "SEa": SEa,
        "matrix": matrix_df,
    }
