"""Revenue time contracts and download eligibility; no invented release dates.

FinMind monthly `date` is a month marker, not an original announcement timestamp.
Strict mode excludes unknown-timestamp records. The explicit lagged scenario is
research only. Raw payloads remain cached so disclosure evidence can be supplied.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MONTH_SOURCE = 'TaiwanStockMonthRevenue'
LEGACY_TIMING = 'vendor_release_date_plus_one_day_not_original_PIT'
EXPLICIT_TIMING = 'EXPLICIT_REVENUE_DISCLOSURE_NOT_REVISION_LEDGER'
ASSUMED_TIMING = 'ASSUMED_REVENUE_MONTH_MARKER_14D_NOT_PIT'
MISSING_TIMING = 'REVENUE_DISCLOSURE_TIME_MISSING_CACHED_NOT_SCORED'


def _stamp(value):
    try:
        t = pd.Timestamp(value)
        if pd.isna(t):
            return pd.NaT
        if t.tzinfo is not None:
            t = t.tz_convert('Asia/Taipei').tz_localize(None)
        return t
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def sanitize_revenue_evidence(evidence, cfg):
    """Migrate old cached month-marker records, without changing other columns.

    User-supplied observations with explicit source+availability and no legacy
    marker flag retain their stated timestamp; the user must supply real evidence.
    A future disclosure stays future. Daily institutional flow is NOT backfilled.
    """
    if evidence is None or evidence.empty or 'revenue_yoy' not in evidence:
        return evidence
    e = evidence.copy()
    for k in ('timing', 'source', 'available_at'):
        if k not in e:
            e[k] = None
    is_month = e.source.fillna('').eq(MONTH_SOURCE) & e.revenue_yoy.notna()
    old = e.timing.fillna('').eq(LEGACY_TIMING)
    unknown = e.timing.fillna('').eq(MISSING_TIMING)
    assumed = e.timing.fillna('').str.startswith('ASSUMED_REVENUE_')
    # Do not guess provenance for unlabelled FinMind month-marker caches.
    unlabelled = is_month & e.timing.fillna('').eq('')
    affected = is_month & (old | unknown | assumed | unlabelled)
    for i in e.index[affected]:
        marker = _stamp(e.at[i, 'date']) if 'date' in e else pd.NaT
        lagged = getattr(cfg, 'financial_timing', 'disclosure_required') == 'lagged_research'
        if lagged and pd.notna(marker):
            available = marker.to_period('M').start_time + pd.Timedelta(days=14)
            known_available = _stamp(e.at[i, 'available_at'])
            if pd.notna(known_available):
                available = max(available, known_available)
            e.at[i, 'available_at'] = str(available)
            e.at[i, 'timing'] = ASSUMED_TIMING
        else:
            # Keep an audit row; never turn absent evidence into a neutral score.
            e.at[i, 'revenue_yoy'] = np.nan
            e.at[i, 'timing'] = MISSING_TIMING
    return e


def monthly_revenue_records(ticker, frame, cfg):
    """Parse raw monthly observations without asserting marker==publication.

    Uses YoY only when a positive prior-year same month is present. Duplicate
    period revisions without a revision ledger remain an explicit limitation.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    needed = {'date', 'revenue', 'revenue_year', 'revenue_month'}
    if not needed.issubset(frame):
        return pd.DataFrame()
    f = frame.sort_values('date').drop_duplicates(['revenue_year', 'revenue_month'], keep='first')
    values = {}
    for r in f.to_dict('records'):
        try:
            value = float(r['revenue'])
            if np.isfinite(value):
                values[(int(r['revenue_year']), int(r['revenue_month']))] = value
        except (ValueError, TypeError):
            continue
    rows = []
    for r in f.to_dict('records'):
        try:
            y, m = int(r['revenue_year']), int(r['revenue_month'])
            old = values.get((y - 1, m)); current = float(r['revenue'])
            period_end = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
            marker = _stamp(r['date'])
            if not old or old <= 0 or not np.isfinite(current) or pd.isna(marker) or marker < period_end:
                continue
            # `create_time` is an ingestion date, not proof of first disclosure.
            dates = [_stamp(r.get(k)) for k in ('announcement_date', 'published_at', 'disclosure_at')]
            dates = [d for d in dates if pd.notna(d) and d >= period_end]
            record = dict(ticker=ticker, date=str(marker.date()), source=MONTH_SOURCE,
                          revenue_yoy=100 * (current / old - 1), revenue_year=y, revenue_month=m,
                          period_end=str(period_end.date()), revision_history_verified=False)
            if dates:
                # Day-only evidence is used from the following calendar day.
                record.update(available_at=str(max(dates) + pd.Timedelta(days=1)), timing=EXPLICIT_TIMING)
            elif getattr(cfg, 'financial_timing', 'disclosure_required') == 'lagged_research':
                available = (period_end + pd.Timedelta(days=1)).normalize() + pd.Timedelta(days=14)
                created = _stamp(r.get('create_time'))
                if pd.notna(created):
                    available = max(available, created + pd.Timedelta(days=1))
                record.update(available_at=str(available), timing=ASSUMED_TIMING)
            else:
                record.update(available_at=None, timing=MISSING_TIMING)
            rows.append(record)
        except (ValueError, TypeError, OverflowError):
            continue
    return pd.DataFrame(rows)
