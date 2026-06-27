import numpy as np

# in this script we will note important functions to use for our key metrics that guide this project.

# Prec@K function
def precision_at_k_percent(y_true, y_score, k_percent):
    """
    Compute precision among the top k_percent highest-scored rows.

    Parameters
    ----------
    y_true : Binary validation labels where 1 indicates active and 0 indicates inactive.

    y_score : Model scores used for ranking validation molecules. Higher scores should
        indicate higher predicted probability of activity.

    k_percent : Fraction of validation molecules to include in the top-ranked subset.
        Use decimal form, e.g. 0.05 for top 5% and 0.10 for top 10%.

    Returns
    -------
    float
        Active fraction among the top-ranked subset.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_true)
    k = int(np.ceil(n * (k_percent / 100)))
    k = max(k, 1)

    ranked_idx = np.argsort(y_score)[::-1]
    top_k_idx = ranked_idx[:k]

    return float(y_true[top_k_idx].mean())

# EF@K function
def enrichment_factor_at_k_percent(y_true, y_score, k_percent):
    """
    Compute EF@K% at top_k_percent.

    EF@K% compares the active fraction in the top K% highest-scored molecules
    against the active fraction in the full validation fold.

    Parameters
    ----------
    y_true : Binary validation labels where 1 indicates active and 0 indicates inactive.

    y_score : Model scores used for ranking validation molecules. Higher scores should
        indicate higher predicted probability of activity.

    k_percent : Fraction of validation molecules to include in the top-ranked subset.
        Use decimal form, e.g. 0.05 for top 5% and 0.10 for top 10%.

    Returns
    -------
    float
        Enrichment factor among the top K% highest-scored validation molecules.
        Values above 1.0 indicate enrichment over random selection from the same
        validation fold. Returns np.nan if the validation fold contains no actives.    
    """
    y_true = np.asarray(y_true)
    active_rate = y_true.mean()

    if active_rate == 0:
        return np.nan

    precision_k = precision_at_k_percent(
        y_true=y_true,
        y_score=y_score,
        k_percent=k_percent,
    )

    return float(precision_k / active_rate)