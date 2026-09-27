"""数据模块：下载、清洗、对齐调整后价格。

所有价格均为 yfinance 调整后收盘价（auto_adjust=True，含分红与拆股调整）。
下载结果缓存在 data/ 目录，重复运行不重复请求网络。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: str, field: str, start: str, end: str,
                tickers=None) -> Path:
    import hashlib
    name = f"{field.lower()}_{start}_{end}"
    if tickers is not None:
        name += "_" + hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    return Path(cache_dir) / f"{name}.csv".replace(":", "-")


def _download_field(tickers, start, end, field: str, cache_dir: str) -> pd.DataFrame:
    """下载某一字段（Close/Volume），带 CSV 缓存（缓存键含 ticker 集合哈希）。"""
    tickers = list(tickers)
    cache = _cache_path(cache_dir, field, start, end, tickers)
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers).issubset(df.columns):
            logger.info("从缓存加载 %s: %s", field, cache)
            return df[tickers]
        logger.info("缓存 %s 缺少部分 ticker，重新下载", cache)

    import yfinance as yf

    logger.info("下载 %s: %s (%s -> %s)", field, tickers, start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance 未返回任何数据: {tickers}")
    df = raw[field]
    if isinstance(df, pd.Series):  # 单 ticker 时降级为 Series
        df = df.to_frame(name=tickers[0])
    df = df.reindex(columns=tickers)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def load_prices(tickers, start, end, cache_dir: str = "data") -> pd.DataFrame:
    """加载调整后收盘价。

    Parameters
    ----------
    tickers : list[str]   代码列表
    start, end : str      日期（end 为开区间）
    cache_dir : str       缓存目录

    Returns
    -------
    pd.DataFrame  宽表，index 为交易日（tz-naive），每列一个 ticker。
    """
    df = _download_field(tickers, start, end, "Close", cache_dir)
    bad = [t for t in tickers if df[t].isna().all()]
    if bad:
        logger.warning("以下 ticker 完全无数据: %s", bad)
    return df


def load_volumes(tickers, start, end, cache_dir: str = "data") -> pd.DataFrame:
    """加载成交量（用于流动性过滤）。"""
    return _download_field(tickers, start, end, "Volume", cache_dir)


def clean_prices(df: pd.DataFrame, ffill_limit: int = 5) -> pd.DataFrame:
    """清洗价格：排序、去重、输出缺失值报告、有限前向填充。

    - 统一为 tz-naive 交易日索引
    - 删除完全无数据的列
    - 缺失值报告写入日志（每只标的缺失数 / 占比 / 有效区间）
    - 前向填充至多 ffill_limit 个交易日，仍有缺失则删除该行
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        logger.warning("丢弃完全无数据的列: %s", empty_cols)
        df = df.drop(columns=empty_cols)

    report = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(3),
        "first_valid": df.apply(lambda s: s.first_valid_index()),
        "last_valid": df.apply(lambda s: s.last_valid_index()),
    })
    logger.info("缺失值报告:\n%s", report.to_string())

    n_before = len(df)
    df = df.ffill(limit=ffill_limit)
    remaining = int(df.isna().sum().sum())
    if remaining:
        logger.warning("ffill(limit=%d) 后仍有 %d 个缺失值，删除对应行", ffill_limit, remaining)
        df = df.dropna()
    logger.info("clean_prices: %d -> %d 行, %d 只标的", n_before, len(df), df.shape[1])
    return df


def align_dates(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """按交易日对内连接两个价格表，删除任一腿缺失的行。"""
    joined = pd.concat([df1, df2], axis=1, join="inner").dropna()
    logger.info("align_dates: %d 个共同交易日", len(joined))
    return joined
