"""
Azure SQL Database access for the finance project.

Uses pymssql, which talks to Azure SQL directly over TDS — no system ODBC
driver (and therefore no admin rights) required. Connection settings are read
from a local, gitignored `db_config.ini` (see `db_config.ini.example`).
Authentication is SQL auth.
"""
import os
import configparser

import pandas as pd
import pymssql

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'db_config.ini')

# --- dbo.share_split schema -------------------------------------------------
# Adjust these two names to match the actual columns in the table.
# Convention: SPLIT_FACTOR_COL stores the ratio such that a 4-for-1 split = 4.0.
SPLIT_DATE_COL = 'split_dt'
SPLIT_FACTOR_COL = 'split_factor'
# ---------------------------------------------------------------------------


def _config() -> configparser.SectionProxy:
    cfg = configparser.ConfigParser()
    if not cfg.read(_CONFIG_PATH):
        raise FileNotFoundError(
            f"DB config not found at {_CONFIG_PATH}. "
            f"Copy db_config.ini.example to db_config.ini and fill it in."
        )
    return cfg['azure_sql']


def get_connection() -> "pymssql.Connection":
    """Open a new pymssql connection to the Azure SQL Database."""
    s = _config()
    return pymssql.connect(
        server=s['server'],
        user=s['username'],
        password=s['password'],
        database=s['database'],
    )


def _rows_to_series(rows) -> pd.Series:
    """Convert [(date, factor), ...] rows into a tz-naive, sorted date->factor Series.

    Pure (no DB) so the parsing contract can be unit-tested. An empty input
    yields an empty float Series, which the caller treats as 'no splits'.
    """
    if not rows:
        return pd.Series(dtype='float64')
    dates = pd.to_datetime([r[0] for r in rows])
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    factors = [float(r[1]) for r in rows]
    return pd.Series(factors, index=dates).sort_index()


def get_share_splits(ticker: str) -> pd.Series:
    """Historic share splits for `ticker` from dbo.share_split.

    Returns a tz-naive Series indexed by split date with the split factor as
    the value (4-for-1 -> 4.0). Returns an empty Series when the ticker has no
    rows, i.e. no splits occurred in the period.
    """
    query = (
        f"SELECT [{SPLIT_DATE_COL}], [{SPLIT_FACTOR_COL}] "
        f"FROM dbo.share_split WHERE ticker = %s"
    )
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, (ticker,))
        rows = cur.fetchall()
    return _rows_to_series(rows)
