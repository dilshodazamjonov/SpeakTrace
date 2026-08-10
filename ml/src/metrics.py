import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats


def _check(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and reject misaligned, empty, or NaN data."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("empty input")
    if np.isnan(t).any() or np.isnan(p).any():
        raise ValueError("NaN in input")
    return t, p


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Root mean squared error, in score units. Primary challenge metric."""
    t, p = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean absolute error. Diagnostic: low MAE with high RMSE means a few large misses."""
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def bias(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean signed error. Positive means the model over-predicts on average."""
    t, p = _check(y_true, y_pred)
    return float(np.mean(p - t))


def pearson(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Linear correlation. Returns NaN when either input is constant."""
    t, p = _check(y_true, y_pred)
    if t.std() == 0 or p.std() == 0:
        return float("nan")
    return float(stats.pearsonr(t, p).statistic)


def spearman(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Rank correlation. Measures ordering only, insensitive to calibration offset."""
    t, p = _check(y_true, y_pred)
    if t.std() == 0 or p.std() == 0:
        return float("nan")
    return float(stats.spearmanr(t, p).statistic)


def within(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, tol: float) -> float:
    """Percentage of predictions falling within +/- tol of the true score, inclusive."""
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(t - p) <= tol) * 100)

def core_metrics(y_true, y_pred) -> dict:
    """
    Summarizes and returns the dict of all the metrics wrapped
    """
    return {
        'n': len(y_true),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'bias': bias(y_true, y_pred),
        'pearson': pearson(y_true, y_pred),
        'spearman': spearman(y_true, y_pred),
        'within_0.5': within(y_true, y_pred, 0.5),
        'within_1.0': within(y_true, y_pred, 1.0)
    }


def evaluate(df: pd.DataFrame) -> dict:
    """
    Validation of parts and no nan in terget cols
    """

    # Validation: 
    if set(df['part']) - {"P1", "P3", "P4", "P5"}:
        raise ValueError("Expected all parts to be one of {P1, P3, P4, P5}")

    if df[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Got nan in neither y_true or y_pred")

    sub = df.groupby('submission_id')[["y_true", "y_pred"]].mean()

    result = {
        "overall": core_metrics(sub["y_true"], sub["y_pred"]),
        "parts": core_metrics(df["y_true"], df["y_pred"]),
    }

    for part in df['part'].unique():

        part_df = df[df['part'] == part][["y_true", "y_pred"]]

        result[part] = core_metrics(part_df["y_true"] , part_df['y_pred'])

    return result
    