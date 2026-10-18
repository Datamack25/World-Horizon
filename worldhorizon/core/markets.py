"""Agent Marchés financiers — indices boursiers nationaux (Yahoo Finance via yfinance),
avec repli sur un ETF pays coté à New York, et indicateurs structurels Banque mondiale.
L'historique quotidien est conservé dans data/markets/<ISO3>.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import MARKETS

# (indice principal, ETF pays de repli, nom de l'indice)
TICKERS = {
    "FRA": ("^FCHI", "EWQ", "CAC 40"), "DEU": ("^GDAXI", "EWG", "DAX"), "GBR": ("^FTSE", "EWU", "FTSE 100"),
    "USA": ("^GSPC", "SPY", "S&P 500"), "CAN": ("^GSPTSE", "EWC", "S&P/TSX"), "JPN": ("^N225", "EWJ", "Nikkei 225"),
    "CHN": ("000001.SS", "MCHI", "SSE Composite"), "HKG": ("^HSI", "EWH", "Hang Seng"), "IND": ("^BSESN", "INDA", "Sensex"),
    "BRA": ("^BVSP", "EWZ", "Ibovespa"), "MEX": ("^MXX", "EWW", "IPC Mexique"), "ARG": ("^MERV", "ARGT", "Merval"),
    "CHL": ("^IPSA", "ECH", "IPSA"), "COL": ("^COLCAP", "GXG", "COLCAP"), "PER": ("^SPBLPGPT", "EPU", "S&P/BVL"),
    "AUS": ("^AXJO", "EWA", "ASX 200"), "NZL": ("^NZ50", "ENZL", "NZX 50"), "KOR": ("^KS11", "EWY", "KOSPI"),
    "TWN": ("^TWII", "EWT", "TAIEX"), "SGP": ("^STI", "EWS", "Straits Times"), "IDN": ("^JKSE", "EIDO", "IDX Composite"),
    "MYS": ("^KLSE", "EWM", "FTSE Bursa Malaysia KLCI"), "THA": ("^SET.BK", "THD", "SET"), "PHL": ("PSEI.PS", "EPHE", "PSEi"),
    "VNM": ("^VNINDEX.VN", "VNM", "VN-Index"), "ISR": ("^TA125.TA", "EIS", "TA-125"), "TUR": ("XU100.IS", "TUR", "BIST 100"),
    "SAU": ("^TASI.SR", "KSA", "Tadawul"), "ARE": ("^DFMGI", "UAE", "DFM"), "QAT": ("^QSI", "QAT", "QE Index"),
    "EGY": ("^CASE30", "EGPT", "EGX 30"), "ZAF": ("^J203.JO", "EZA", "JSE All Share"), "NGA": ("^NGSE", "NGE", "NGX ASI"),
    "MAR": ("^MASI", None, "MASI"), "KEN": ("^NSE20", None, "NSE 20"),
    "NLD": ("^AEX", "EWN", "AEX"), "BEL": ("^BFX", "EWK", "BEL 20"), "ESP": ("^IBEX", "EWP", "IBEX 35"),
    "ITA": ("FTSEMIB.MI", "EWI", "FTSE MIB"), "CHE": ("^SSMI", "EWL", "SMI"), "SWE": ("^OMX", "EWD", "OMX Stockholm 30"),
    "NOR": ("OSEBX.OL", "NORW", "OBX"), "DNK": ("^OMXC25", "EDEN", "OMX Copenhague 25"), "FIN": ("^OMXH25", "EFNL", "OMX Helsinki 25"),
    "AUT": ("^ATX", "EWO", "ATX"), "IRL": ("^ISEQ", "EIRL", "ISEQ"), "PRT": ("PSI20.LS", "PGAL", "PSI"),
    "POL": ("WIG20.WA", "EPOL", "WIG20"), "GRC": ("GD.AT", "GREK", "ATHEX"), "HUN": ("^BUX.BD", None, "BUX"),
    "CZE": ("^PX", None, "PX"), "RUS": ("IMOEX.ME", None, "MOEX"), "PAK": ("^KSE", "PAK", "KSE 100"),
}


def _stats(close: pd.Series) -> dict:
    close = close.dropna()
    if len(close) < 30:
        return {}
    last = float(close.iloc[-1])

    def perf(days):
        d = close.index[-1] - pd.Timedelta(days=days)
        past = close[close.index <= d]
        return float(last / past.iloc[-1] - 1) * 100 if len(past) else np.nan
    rets = np.log(close).diff().dropna()
    vol = float(rets[-252:].std() * np.sqrt(252) * 100)
    dd = float((close / close.cummax() - 1).min() * 100)
    return dict(dernier=last, date=str(close.index[-1].date()), perf_1m=perf(30), perf_1a=perf(365),
                perf_5a=perf(5 * 365), volatilite_1a=vol, drawdown_max=dd)


def fetch_market(iso3: str, period: str = "10y") -> dict | None:
    if iso3 not in TICKERS:
        return None
    idx, etf, name = TICKERS[iso3]
    path = MARKETS / f"{iso3}.csv"
    close, used, from_cache = None, None, False
    try:
        import yfinance as yf
        for t in [x for x in (idx, etf) if x]:
            h = yf.Ticker(t).history(period=period, auto_adjust=True)
            if h is not None and len(h) > 30:
                close = h["Close"]; close.index = pd.to_datetime(close.index).tz_localize(None); used = t
                break
    except Exception:
        close = None
    if close is not None:
        df = close.rename("close").to_frame(); df["ticker"] = used
        if path.exists():
            old = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        df.index.name = "date"
        df.to_csv(path)
    elif path.exists():
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        close, used, from_cache = df["close"], df["ticker"].iloc[-1], True
    else:
        return None
    s = _stats(close)
    return dict(iso3=iso3, ticker=used, nom=name if used == idx else f"ETF {used} (repli)", serie=close, **s,
                source=f"Yahoo Finance — {used}", hors_ligne=from_cache)


def refresh_all(log=print) -> pd.DataFrame:
    rows = []
    for iso in TICKERS:
        m = fetch_market(iso, period="2y")
        if m and m.get("dernier"):
            rows.append({k: v for k, v in m.items() if k != "serie"})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(MARKETS / "_summary.csv", index=False)
    log(f"Marchés : {len(df)}/{len(TICKERS)} indices mis à jour")
    return df
