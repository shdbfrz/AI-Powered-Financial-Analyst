"""
Stationarity analysis for `ai/models/time_series/`.

Classical models (ARIMA/SARIMA) assume a stationary series (constant mean,
variance, and autocovariance over time). This module runs the two
complementary tests the Sprint 4 spec requires — ADF and KPSS — and derives
a differencing order `d` from their combined verdict, because the two tests
have opposite null hypotheses and neither alone is reliable:

- **ADF** (Augmented Dickey-Fuller): H0 = series has a unit root
  (non-stationary). A *low* p-value (< alpha) rejects H0, i.e. the series
  IS stationary.
- **KPSS**: H0 = series IS stationary. A *low* p-value (< alpha) rejects H0,
  i.e. the series is NOT stationary.

Combining them (Kwiatkowski et al., 1992 recommend exactly this pairing)
gives four cases instead of one test's two:

| ADF says stationary | KPSS says stationary | Verdict                          |
|---|---|---|
| Yes | Yes | Stationary — no differencing needed |
| No  | No  | Non-stationary — difference and re-test |
| No  | Yes | Trend-stationary — de-trend (differencing still works) |
| Yes | No  | Difference-stationary — differencing needed despite ADF |
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import skew
from statsmodels.tsa.stattools import adfuller, kpss

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StationarityTestResult:
    test_name: str
    statistic: float
    p_value: float
    critical_values: dict
    is_stationary: bool
    significance_level: float

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(float(self.statistic), 6),
            "p_value": round(float(self.p_value), 6),
            "critical_values": {k: round(float(v), 6) for k, v in self.critical_values.items()},
            "is_stationary": self.is_stationary,
            "significance_level": self.significance_level,
        }


@dataclass
class StationarityReport:
    adf: StationarityTestResult
    kpss: StationarityTestResult
    combined_is_stationary: bool
    verdict: str
    recommended_differencing_order: int
    recommended_log_transform: bool
    skewness: float
    differenced_reports: list["StationarityReport"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "adf": self.adf.to_dict(),
            "kpss": self.kpss.to_dict(),
            "combined_is_stationary": self.combined_is_stationary,
            "verdict": self.verdict,
            "recommended_differencing_order": self.recommended_differencing_order,
            "recommended_log_transform": self.recommended_log_transform,
            "skewness": round(float(self.skewness), 6),
        }


def run_adf_test(series: pd.Series, *, significance_level: float = 0.05) -> StationarityTestResult:
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary)."""
    clean = series.dropna()
    statistic, p_value, _, _, critical_values, _ = adfuller(clean, autolag="AIC")
    return StationarityTestResult(
        test_name="ADF",
        statistic=statistic,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=p_value < significance_level,
        significance_level=significance_level,
    )


def run_kpss_test(series: pd.Series, *, significance_level: float = 0.05) -> StationarityTestResult:
    """KPSS test. H0: series IS (trend-)stationary."""
    clean = series.dropna()
    import warnings

    with warnings.catch_warnings():
        # statsmodels warns when the KPSS statistic falls outside the table
        # of tabulated p-values; the returned p-value is still a valid
        # (clamped) estimate, so we surface it rather than error out.
        warnings.simplefilter("ignore")
        statistic, p_value, _, critical_values = kpss(clean, regression="c", nlags="auto")
    return StationarityTestResult(
        test_name="KPSS",
        statistic=statistic,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=p_value >= significance_level,
        significance_level=significance_level,
    )


def _verdict(adf_stationary: bool, kpss_stationary: bool) -> tuple[str, bool]:
    if adf_stationary and kpss_stationary:
        return "stationary", True
    if not adf_stationary and not kpss_stationary:
        return "non_stationary", False
    if not adf_stationary and kpss_stationary:
        return "trend_stationary", False
    return "difference_stationary", False


def analyze_stationarity(
    series: pd.Series,
    *,
    config: TimeSeriesConfig = DEFAULT_TS_CONFIG,
    _current_order: int = 0,
) -> StationarityReport:
    """Run ADF + KPSS, and if the combined verdict is non-stationary,
    recursively difference (up to `config.max_differencing_order` times) and
    re-test, recording each round in `differenced_reports`.

    Never raises on failing to reach full stationarity — it logs and returns
    the best-effort order at the configured maximum instead, since a caller
    (the training pipeline) should still be able to proceed with (p, d, q)
    search using that order rather than hard-failing the whole run.
    """
    adf_result = run_adf_test(series, significance_level=config.adf_significance_level)
    kpss_result = run_kpss_test(series, significance_level=config.kpss_significance_level)
    verdict, is_stationary = _verdict(adf_result.is_stationary, kpss_result.is_stationary)

    series_skew = float(skew(series.dropna()))
    recommend_log = abs(series_skew) > config.log_transform_skew_threshold and (series.dropna() > 0).all()

    report = StationarityReport(
        adf=adf_result,
        kpss=kpss_result,
        combined_is_stationary=is_stationary,
        verdict=verdict,
        recommended_differencing_order=_current_order,
        recommended_log_transform=recommend_log,
        skewness=series_skew,
    )

    logger.info(
        "stationarity test complete",
        extra={
            "order": _current_order,
            "verdict": verdict,
            "adf_p": round(adf_result.p_value, 4),
            "kpss_p": round(kpss_result.p_value, 4),
        },
    )

    if is_stationary or _current_order >= config.max_differencing_order:
        if not is_stationary:
            logger.info(
                "reached max differencing order without full stationarity; proceeding with best-effort order",
                extra={"order": _current_order, "verdict": verdict},
            )
        return report

    differenced = series.diff().dropna()
    next_report = analyze_stationarity(differenced, config=config, _current_order=_current_order + 1)
    report.differenced_reports = [next_report, *next_report.differenced_reports]
    next_report.differenced_reports = []
    report.recommended_differencing_order = (
        report.differenced_reports[-1].recommended_differencing_order
        if report.differenced_reports
        else _current_order
    )
    return report


def difference_series(series: pd.Series, order: int) -> pd.Series:
    """Apply `order` rounds of first-differencing. `order=0` returns the
    series unchanged (copy)."""
    if order < 0:
        raise ValueError(f"differencing order must be >= 0, got {order}")
    result = series.copy()
    for _ in range(order):
        result = result.diff()
    return result.dropna()


def log_transform(series: pd.Series) -> pd.Series:
    """Natural-log transform, used to stabilize variance before differencing
    when `StationarityReport.recommended_log_transform` is True. Requires an
    all-positive series (prices always are, post NaN-fill)."""
    if (series.dropna() <= 0).any():
        raise ValueError("log_transform requires a strictly positive series")
    return np.log(series)


def inverse_log_transform(series: pd.Series) -> pd.Series:
    return np.exp(series)