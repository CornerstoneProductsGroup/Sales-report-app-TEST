
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Reuse ingestion helpers from the current (legacy) app where possible.
from modules.app_core import read_weekly_workbook, parse_date_range_from_filename

APP_TITLE = "Cornerstone Sales Intelligence"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_VENDOR_MAP = DATA_DIR / "vendor_map.xlsx"
DEFAULT_STORE_CSV = DATA_DIR / "sales_store.csv"

# -----------------------------
# Normalization helpers
# -----------------------------
_RETAILER_ALIASES = {
    "home depot": "Depot",
    "the home depot": "Depot",
    "depot": "Depot",
    "lowe's": "Lowes",
    "lowes": "Lowes",
    "tractor supply": "Tractor Supply",
    "tsc": "Tractor Supply",
    "home depot canada": "Home Depot Canada",
    "ace": "Ace",
    "amazon": "Amazon",
    "walmart": "Walmart",
    "zoro": "Zoro",
    "orgill": "Orgill",
}

def norm_retailer(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    k = s.lower()
    return _RETAILER_ALIASES.get(k, s)

def norm_sku(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(" ", "")
    return s

# -----------------------------
# Storage
# -----------------------------
BASE_COLUMNS = ["Retailer","Vendor","SKU","Units","Price","Sales","StartDate","EndDate","SourceFile"]

def load_store() -> pd.DataFrame:
    if DEFAULT_STORE_CSV.exists():
        df = pd.read_csv(DEFAULT_STORE_CSV)
    else:
        df = pd.DataFrame(columns=["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"])
    # If it's legacy shape (no Vendor/Price/Sales), keep and enrich later.
    for c in ["StartDate","EndDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Retailer" in df.columns:
        df["Retailer"] = df["Retailer"].map(norm_retailer)
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].map(norm_sku)
    return df

def save_store(df: pd.DataFrame) -> None:
    # Persist in the legacy shape to stay compatible with the existing app if needed.
    keep = df.copy()
    # Ensure these exist
    for c in ["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]:
        if c not in keep.columns:
            keep[c] = np.nan
    keep = keep[["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]].copy()
    keep.to_csv(DEFAULT_STORE_CSV, index=False)

def load_vendor_map() -> pd.DataFrame:
    if not DEFAULT_VENDOR_MAP.exists():
        return pd.DataFrame(columns=["Retailer","SKU","Price","Vendor"])
    df = pd.read_excel(DEFAULT_VENDOR_MAP, sheet_name=0, engine="openpyxl")
    # Minimal standardization
    for c in ["Retailer","SKU","Vendor"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "Retailer" in df.columns:
        df["Retailer"] = df["Retailer"].map(norm_retailer)
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].map(norm_sku)
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df[["Retailer","SKU","Price","Vendor"]].copy()

def enrich_sales(df_raw: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    """Return a fully-enriched fact table with Vendor, Price, Sales, and a weekly key."""
    df = df_raw.copy()
    # Ensure base columns exist
    for c in ["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Retailer"] = df["Retailer"].map(norm_retailer)
    df["SKU"] = df["SKU"].map(norm_sku)
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0.0)

    # Merge vendor map (Retailer+SKU)
    m = vm.copy()
    df = df.merge(m, on=["Retailer","SKU"], how="left", suffixes=("","_map"))
    # Use UnitPrice if present, else map Price
    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice"), errors="coerce")
    df["Price"] = np.where(df["UnitPrice"].notna(), df["UnitPrice"], df["Price"])
    df["Sales"] = df["Units"] * df["Price"].fillna(0.0)

    df["Vendor"] = df["Vendor"].fillna("Unknown").astype(str).str.strip()
    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
    df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")
    # Weekly key: use EndDate; fallback to StartDate
    df["WeekEnd"] = df["EndDate"].fillna(df["StartDate"])
    df["WeekEnd"] = pd.to_datetime(df["WeekEnd"], errors="coerce")
    df["Year"] = df["WeekEnd"].dt.year
    # Display label like "2026-01-05 / 2026-01-09"
    df["WeekLabel"] = df.apply(
        lambda r: (f"{r['StartDate'].date()} / {r['EndDate'].date()}" if pd.notna(r["StartDate"]) and pd.notna(r["EndDate"]) else (str(r["WeekEnd"].date()) if pd.notna(r["WeekEnd"]) else "")),
        axis=1
    )
    return df

# -----------------------------
# Period selection
# -----------------------------
@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp

def _safe_max_ts(s: pd.Series) -> Optional[pd.Timestamp]:
    s2 = pd.to_datetime(s, errors="coerce")
    s2 = s2.dropna()
    return None if s2.empty else s2.max()

def pick_period(df: pd.DataFrame, mode: str, n_weeks: int = 8) -> Optional[Period]:
    """Choose the current (A) period based on mode using df's max WeekEnd as anchor."""
    anchor = _safe_max_ts(df.get("WeekEnd", pd.Series(dtype="datetime64[ns]")))
    if anchor is None:
        return None

    if mode.startswith("Last "):
        # "Last 4 weeks", etc.
        weeks = int(re.findall(r"\d+", mode)[0])
        start = anchor - pd.Timedelta(days=(7*weeks - 1))
        return Period(start=start.normalize(), end=anchor.normalize())
    if mode == "Week (latest)":
        start = anchor - pd.Timedelta(days=6)
        return Period(start=start.normalize(), end=anchor.normalize())
    if mode == "YTD":
        start = pd.Timestamp(year=anchor.year, month=1, day=1)
        return Period(start=start, end=anchor.normalize())
    return None

def period_prev_same_length(p: Period) -> Period:
    length = (p.end - p.start).days + 1
    end = p.start - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=length-1)
    return Period(start=start.normalize(), end=end.normalize())

def period_yoy(p: Period) -> Period:
    # Shift by 1 year (approx) using DateOffset to handle leap years
    start = (p.start - pd.DateOffset(years=1)).normalize()
    end = (p.end - pd.DateOffset(years=1)).normalize()
    return Period(start=start, end=end)

def filter_by_period(df: pd.DataFrame, p: Period) -> pd.DataFrame:
    w = pd.to_datetime(df["WeekEnd"], errors="coerce")
    return df[(w >= p.start) & (w <= p.end)].copy()

# -----------------------------
# KPI + analytics engines
# -----------------------------
def calc_kpis(df: pd.DataFrame) -> Dict[str, float]:
    sales = float(df["Sales"].sum())
    units = float(df["Units"].sum())
    asp = float(sales / units) if units else 0.0
    active_skus = int(df.loc[df["Sales"] > 0, "SKU"].nunique())
    active_retailers = int(df.loc[df["Sales"] > 0, "Retailer"].nunique())
    active_vendors = int(df.loc[df["Sales"] > 0, "Vendor"].nunique())
    return {
        "Sales": sales,
        "Units": units,
        "ASP": asp,
        "Active SKUs": active_skus,
        "Active Retailers": active_retailers,
        "Active Vendors": active_vendors,
    }

def calc_delta(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, va in a.items():
        vb = b.get(k, 0.0)
        out[k] = va - vb
    return out

def pct_change(cur: float, prev: float) -> float:
    if prev == 0:
        return np.nan if cur == 0 else np.inf
    return (cur - prev) / prev

def drivers(df_a: pd.DataFrame, df_b: pd.DataFrame, level: str) -> pd.DataFrame:
    """Top drivers by contribution (Sales_A - Sales_B)."""
    g = [level]
    a = df_a.groupby(g, as_index=False).agg(Sales_A=("Sales","sum"), Units_A=("Units","sum"))
    b = df_b.groupby(g, as_index=False).agg(Sales_B=("Sales","sum"), Units_B=("Units","sum"))
    m = a.merge(b, on=g, how="outer").fillna(0.0)
    m["Sales_Δ"] = m["Sales_A"] - m["Sales_B"]
    m["Units_Δ"] = m["Units_A"] - m["Units_B"]
    total = float(m["Sales_Δ"].sum())
    m["Contribution_%"] = np.where(total != 0, m["Sales_Δ"] / total, 0.0)
    m = m.sort_values("Sales_Δ", ascending=False)
    return m

def weekly_series(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    s = df.groupby(by + ["WeekEnd"], as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
    s = s.sort_values("WeekEnd")
    return s

def trend_slope(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    x = np.arange(values.size, dtype=float)
    y = values.astype(float)
    if np.all(np.isnan(y)) or np.all(y == 0):
        return 0.0
    # Replace nans with 0 for stability
    y = np.nan_to_num(y, nan=0.0)
    return float(np.polyfit(x, y, 1)[0])

def classify_trend(series: pd.Series, min_weeks: int = 8) -> Tuple[str, float, int, int]:
    y = np.array(series.values, dtype=float)
    slope = trend_slope(y)
    weeks_up = int(np.sum(np.diff(y) > 0))
    weeks_down = int(np.sum(np.diff(y) < 0))
    # threshold scales with median magnitude to avoid tiny-noise labeling
    med = float(np.median(np.abs(y))) if y.size else 0.0
    thr = max(1.0, 0.03 * med)  # 3% of median, min 1
    if slope > thr and weeks_up >= max(2, (min_weeks//2)):
        return ("Increasing", slope, weeks_up, weeks_down)
    if slope < -thr and weeks_down >= max(2, (min_weeks//2)):
        return ("Declining", slope, weeks_up, weeks_down)
    return ("Flat", slope, weeks_up, weeks_down)

def momentum_score(series: pd.Series) -> float:
    y = np.array(series.values, dtype=float)
    if y.size < 3:
        return 0.0
    y = np.nan_to_num(y, nan=0.0)
    slope = trend_slope(y)
    # normalize slope vs median
    med = np.median(np.abs(y)) if np.any(y) else 1.0
    trend = np.clip((slope / max(1.0, med)) * 30.0, -30.0, 30.0)

    # acceleration: last 2 vs prior 4
    recent = np.mean(y[-2:]) if y.size >= 2 else y[-1]
    prior = np.mean(y[-6:-2]) if y.size >= 6 else (np.mean(y[:-2]) if y.size > 2 else y[0])
    accel = 0.0
    if prior != 0:
        accel = np.clip(((recent - prior) / abs(prior)) * 20.0, -20.0, 20.0)
    else:
        accel = 20.0 if recent > 0 else 0.0

    # consistency
    up = np.sum(np.diff(y) > 0)
    denom = max(1, y.size - 1)
    cons = np.clip((up / denom) * 50.0, 0.0, 50.0)

    score = float(np.clip(trend + accel + cons, 0.0, 100.0))
    return score

def momentum_label(score: float) -> str:
    if score >= 80:
        return "Strong Up"
    if score >= 60:
        return "Up"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Down"
    return "Strong Down"

def build_momentum(df_hist: pd.DataFrame, group_level: str, lookback_weeks: int = 8) -> pd.DataFrame:
    """Compute momentum for each group using last N weeks."""
    s = weekly_series(df_hist, [group_level])
    # limit to last N weeks per group
    out_rows = []
    for key, g in s.groupby(group_level):
        g = g.sort_values("WeekEnd")
        g = g.tail(lookback_weeks)
        score = momentum_score(g["Sales"])
        trend, slope, wu, wd = classify_trend(g["Sales"], min_weeks=min(lookback_weeks, len(g)))
        out_rows.append({
            group_level: key,
            "Momentum": score,
            "Momentum Label": momentum_label(score),
            "Trend": trend,
            "Slope": slope,
            "Weeks Up": wu,
            "Weeks Down": wd,
            "Sales (lookback)": float(g["Sales"].sum()),
            "Units (lookback)": float(g["Units"].sum()),
        })
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["Momentum","Sales (lookback)"], ascending=[False, False])
    return out

# -----------------------------
# Newness / lifecycle
# -----------------------------
def first_sale_ever(df_all: pd.DataFrame, p: Period) -> pd.DataFrame:
    d = df_all[df_all["Sales"] > 0].copy()
    if d.empty:
        return d
    first = d.groupby("SKU", as_index=False).agg(FirstWeek=("WeekEnd","min"), FirstRetailer=("Retailer","first"), FirstVendor=("Vendor","first"))
    in_period = first[(first["FirstWeek"] >= p.start) & (first["FirstWeek"] <= p.end)].copy()
    return in_period.sort_values("FirstWeek")

def new_placement(df_all: pd.DataFrame, p: Period) -> pd.DataFrame:
    d = df_all[df_all["Sales"] > 0].copy()
    if d.empty:
        return d
    fr = d.groupby(["SKU","Retailer"], as_index=False).agg(FirstWeek=("WeekEnd","min"), Vendor=("Vendor","first"))
    in_period = fr[(fr["FirstWeek"] >= p.start) & (fr["FirstWeek"] <= p.end)].copy()
    # Exclude those that are also first sale ever
    first_sku = d.groupby("SKU", as_index=False).agg(SKUFirst=("WeekEnd","min"))
    in_period = in_period.merge(first_sku, on="SKU", how="left")
    in_period = in_period[in_period["SKUFirst"] < in_period["FirstWeek"]]
    return in_period.sort_values("FirstWeek")

def reactivated(df_all: pd.DataFrame, p: Period, dormant_weeks: int = 8) -> pd.DataFrame:
    # A SKU with sales in current period, and previously had a gap of >= dormant_weeks with no sales.
    d = df_all.copy()
    d = d.sort_values("WeekEnd")
    # build weekly sales per SKU
    s = weekly_series(d, ["SKU"])
    if s.empty:
        return pd.DataFrame(columns=["SKU","ReactivatedWeek","DormantWeeks"])
    out=[]
    for sku, g in s.groupby("SKU"):
        g = g.sort_values("WeekEnd")
        # mark weeks with sales>0
        g["HasSales"] = g["Sales"] > 0
        # consider weeks within current period
        cur = g[(g["WeekEnd"] >= p.start) & (g["WeekEnd"] <= p.end) & (g["HasSales"])]
        if cur.empty:
            continue
        # find most recent sale before current period
        prev_sales = g[(g["WeekEnd"] < p.start) & (g["HasSales"])]
        if prev_sales.empty:
            continue
        last_prev = prev_sales["WeekEnd"].max()
        first_cur = cur["WeekEnd"].min()
        gap_weeks = int((first_cur - last_prev).days // 7)  # approximate
        if gap_weeks >= dormant_weeks:
            out.append({"SKU": sku, "ReactivatedWeek": first_cur, "DormantWeeks": gap_weeks})
    return pd.DataFrame(out).sort_values("ReactivatedWeek") if out else pd.DataFrame(columns=["SKU","ReactivatedWeek","DormantWeeks"])

def lifecycle_table(df_all: pd.DataFrame, p: Period, lookback_weeks: int = 8) -> pd.DataFrame:
    # Stage per SKU using full history but relative to current period.
    s = weekly_series(df_all, ["SKU"])
    if s.empty:
        return pd.DataFrame(columns=["SKU","Stage","Momentum","Sales (lookback)"])
    # universe and anchor = period end
    anchor = p.end
    lb_start = anchor - pd.Timedelta(days=(7*lookback_weeks - 1))
    out=[]
    first_week = df_all[df_all["Sales"] > 0].groupby("SKU")["WeekEnd"].min()
    last_week = df_all[df_all["Sales"] > 0].groupby("SKU")["WeekEnd"].max()

    # placements
    fr = df_all[df_all["Sales"] > 0].groupby(["SKU","Retailer"])["WeekEnd"].min().reset_index().rename(columns={"WeekEnd":"FirstWeekRetailer"})

    for sku, g in s.groupby("SKU"):
        g = g.sort_values("WeekEnd")
        # window
        gw = g[(g["WeekEnd"] >= lb_start) & (g["WeekEnd"] <= anchor)].copy()
        mom = momentum_score(gw["Sales"]) if not gw.empty else 0.0
        sales_lb = float(gw["Sales"].sum()) if not gw.empty else 0.0

        fw = first_week.get(sku, pd.NaT)
        lw = last_week.get(sku, pd.NaT)

        stage = "Mature"
        # Launch: first sale ever in current period
        if pd.notna(fw) and (fw >= p.start) and (fw <= p.end):
            stage = "Launch"
        else:
            # Dormant: no sale in lookback window
            if pd.isna(lw) or lw < lb_start:
                stage = "Dormant"
            else:
                # Growth/Decline based on trend classification in lookback
                if not gw.empty:
                    trend, slope, wu, wd = classify_trend(gw["Sales"], min_weeks=min(lookback_weeks, len(gw)))
                    if trend == "Increasing":
                        stage = "Growth"
                    elif trend == "Declining":
                        stage = "Decline"
                    else:
                        stage = "Mature"

        out.append({"SKU": sku, "Stage": stage, "Momentum": mom, "Sales (lookback)": sales_lb})
    out_df = pd.DataFrame(out)
    if not out_df.empty:
        out_df = out_df.sort_values(["Stage","Momentum","Sales (lookback)"], ascending=[True, False, False])
    return out_df

# -----------------------------
# Opportunity detector
# -----------------------------
def opportunity_detector(df_all: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame, p: Period) -> Dict[str, pd.DataFrame]:
    retailers = sorted(df_all["Retailer"].dropna().unique().tolist())
    if not retailers:
        retailers = []
    # Momentum per SKU on full history up to period end
    anchor = p.end
    df_hist = df_all[df_all["WeekEnd"] <= anchor].copy()
    mom_sku = build_momentum(df_hist, "SKU", lookback_weeks=8)
    mom_sku = mom_sku.set_index("SKU") if not mom_sku.empty else pd.DataFrame()

    # Current period retailer presence per SKU
    cur = df_a.groupby(["SKU","Retailer"], as_index=False).agg(Sales=("Sales","sum"))
    cur_pos = cur[cur["Sales"] > 0].copy()

    # 1) High momentum + low distribution
    hml = []
    if not cur_pos.empty:
        selling_counts = cur_pos.groupby("SKU")["Retailer"].nunique()
        for sku, cnt in selling_counts.items():
            score = float(mom_sku.loc[sku, "Momentum"]) if (isinstance(mom_sku, pd.DataFrame) and sku in mom_sku.index) else 0.0
            if score >= 80 and cnt <= 1:
                hml.append({"SKU": sku, "Momentum": score, "Retailers Selling": int(cnt)})
    high_mom_low_dist = pd.DataFrame(hml).sort_values(["Momentum","Retailers Selling"], ascending=[False, True]) if hml else pd.DataFrame(columns=["SKU","Momentum","Retailers Selling"])

    # 2) Under-distributed opportunities (missing retailer where vendor is active)
    under = []
    # vendor activity by retailer in current period
    vend_ret = df_a.groupby(["Vendor","Retailer"], as_index=False).agg(Sales=("Sales","sum"))
    vend_ret = vend_ret[vend_ret["Sales"] > 0]
    vend_active = set(zip(vend_ret["Vendor"], vend_ret["Retailer"]))

    sku_vendor = df_all.groupby("SKU", as_index=False).agg(Vendor=("Vendor","first"))
    sku_to_vendor = dict(zip(sku_vendor["SKU"], sku_vendor["Vendor"]))
    if not df_a.empty:
        sku_rets = cur_pos.groupby("SKU")["Retailer"].apply(set).to_dict()
        for sku, sold_set in sku_rets.items():
            score = float(mom_sku.loc[sku, "Momentum"]) if (isinstance(mom_sku, pd.DataFrame) and sku in mom_sku.index) else 0.0
            if score < 60:
                continue
            vendor = sku_to_vendor.get(sku, "Unknown")
            for r in retailers:
                if r in sold_set:
                    continue
                if (vendor, r) in vend_active:
                    under.append({"SKU": sku, "Vendor": vendor, "Missing Retailer": r, "Momentum": score})
    under_df = pd.DataFrame(under).sort_values(["Momentum"], ascending=False) if under else pd.DataFrame(columns=["SKU","Vendor","Missing Retailer","Momentum"])

    # 3) Retailer growth gaps (vendor-level by default)
    a = df_a.groupby(["Vendor","Retailer"], as_index=False).agg(Sales_A=("Sales","sum"))
    b = df_b.groupby(["Vendor","Retailer"], as_index=False).agg(Sales_B=("Sales","sum"))
    m = a.merge(b, on=["Vendor","Retailer"], how="outer").fillna(0.0)
    # compute growth %
    m["Growth_%"] = m.apply(lambda r: pct_change(float(r["Sales_A"]), float(r["Sales_B"])), axis=1)
    # For each vendor, compute range across retailers (max-min)
    gaps=[]
    for vendor, g in m.groupby("Vendor"):
        if g.empty:
            continue
        # require some meaningful base
        if float(g["Sales_A"].sum()) < 500:
            continue
        # ignore inf/nan
        gg = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["Growth_%"])
        if gg.empty:
            continue
        mx = float(gg["Growth_%"].max())
        mn = float(gg["Growth_%"].min())
        if (mx - mn) >= 0.30:  # 30pp gap
            # show top and bottom retailer
            top = gg.loc[gg["Growth_%"].idxmax()]
            bot = gg.loc[gg["Growth_%"].idxmin()]
            gaps.append({
                "Vendor": vendor,
                "Best Retailer": top["Retailer"],
                "Best Growth %": top["Growth_%"],
                "Worst Retailer": bot["Retailer"],
                "Worst Growth %": bot["Growth_%"],
                "Gap (pp)": (mx - mn),
                "Sales A": float(g["Sales_A"].sum()),
            })
    gaps_df = pd.DataFrame(gaps).sort_values("Gap (pp)", ascending=False) if gaps else pd.DataFrame(columns=["Vendor","Best Retailer","Best Growth %","Worst Retailer","Worst Growth %","Gap (pp)","Sales A"])

    return {
        "High Momentum / Low Distribution": high_mom_low_dist,
        "Under-distributed Opportunities": under_df,
        "Retailer Growth Gaps": gaps_df,
    }

# -----------------------------
# UI helpers
# -----------------------------
def money(x: float) -> str:
    return f"${x:,.0f}"

def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    if x == np.inf:
        return "∞"
    if x == -np.inf:
        return "-∞"
    return f"{x*100:,.1f}%"

def kpi_card(label: str, value: str, delta: Optional[str] = None):
    st.markdown(
        f"""
        <div style="border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:14px 14px; background:white;">
            <div style="font-size:12px; color:rgba(0,0,0,0.55); font-weight:600; letter-spacing:0.02em;">{label}</div>
            <div style="font-size:28px; font-weight:800; margin-top:6px;">{value}</div>
            <div style="font-size:12px; color:rgba(0,0,0,0.55); margin-top:2px;">{delta or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_df(df: pd.DataFrame, height: int = 320):
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)

# -----------------------------
# Main app
# -----------------------------
def run_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    vm = load_vendor_map()
    store = load_store()

    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload weekly sales workbook (.xlsx)", type=["xlsx"])
        year = st.number_input("Year hint (for filename parsing)", min_value=2010, max_value=2100, value=date.today().year, step=1)
        if st.button("Ingest upload", disabled=(up is None)):
            if up is not None:
                raw = read_weekly_workbook(up, int(year))
                # enrich now so we can persist UnitPrice (legacy) but also show computed
                new = enrich_sales(raw, vm)
                # Persist in legacy shape
                merged = pd.concat([store, raw], ignore_index=True)
                save_store(merged)
                st.success(f"Ingested {len(raw):,} rows from {getattr(up,'name','upload.xlsx')}.")
                store = load_store()

        st.divider()
        st.header("Filters")

        scope = st.selectbox("Scope", ["All", "Retailer", "Vendor", "SKU"], index=0)
        # We build an enriched df first (so Vendor exists)
        df_all = enrich_sales(store, vm)
        # Scope pickers
        scope_pick = None
        if scope == "Retailer":
            scope_pick = st.multiselect("Retailer(s)", options=sorted(df_all["Retailer"].dropna().unique()), default=[])
        elif scope == "Vendor":
            scope_pick = st.multiselect("Vendor(s)", options=sorted(df_all["Vendor"].dropna().unique()), default=[])
        elif scope == "SKU":
            scope_pick = st.multiselect("SKU(s)", options=sorted(df_all["SKU"].dropna().unique()), default=[])

        timeframe = st.selectbox("Timeframe", ["Week (latest)", "Last 4 weeks", "Last 8 weeks", "Last 13 weeks", "Last 26 weeks", "Last 52 weeks", "YTD"], index=2)
        compare_mode = st.selectbox("Compare", ["None", "Prior period (same length)", "YoY (same dates)"], index=1)

        min_sales = st.number_input("Min Sales ($) for lists", min_value=0.0, value=0.0, step=100.0)
        min_units = st.number_input("Min Units for lists", min_value=0.0, value=0.0, step=10.0)

        driver_level = st.selectbox("Driver Level", ["SKU", "Vendor", "Retailer"], index=0)
        show_full_history_lifecycle = st.toggle("Lifecycle uses full history", value=True)

    # Apply scope filter
    df_scope = df_all.copy()
    if scope == "Retailer" and scope_pick:
        df_scope = df_scope[df_scope["Retailer"].isin(scope_pick)]
    elif scope == "Vendor" and scope_pick:
        df_scope = df_scope[df_scope["Vendor"].isin(scope_pick)]
    elif scope == "SKU" and scope_pick:
        df_scope = df_scope[df_scope["SKU"].isin(scope_pick)]

    # Choose current period
    pA = pick_period(df_scope, timeframe)
    if pA is None:
        st.info("Upload or ingest data to begin.")
        return

    dfA = filter_by_period(df_scope, pA)
    if compare_mode == "None":
        pB = None
        dfB = dfA.iloc[0:0].copy()
    elif compare_mode.startswith("Prior"):
        pB = period_prev_same_length(pA)
        dfB = filter_by_period(df_scope, pB)
    else:
        pB = period_yoy(pA)
        dfB = filter_by_period(df_scope, pB)

    # KPI compute
    kA = calc_kpis(dfA)
    kB = calc_kpis(dfB) if pB is not None else {k:0.0 for k in kA.keys()}

    # Newness metrics must use history; use df_all unless scope filtered is desired
    df_hist_for_new = df_all.copy()
    if scope == "Retailer" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["Retailer"].isin(scope_pick)]
    elif scope == "Vendor" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["Vendor"].isin(scope_pick)]
    elif scope == "SKU" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["SKU"].isin(scope_pick)]

    first_ever = first_sale_ever(df_hist_for_new, pA)
    placements = new_placement(df_hist_for_new, pA)

    # Layout
    st.markdown(
        f"<div style='color:rgba(0,0,0,0.55); font-weight:600;'>Period: {pA.start.date()} → {pA.end.date()}"
        + (f" &nbsp;&nbsp;|&nbsp;&nbsp; Compare: {pB.start.date()} → {pB.end.date()}" if pB is not None else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # 0) Intelligence summary
    sales_delta = kA["Sales"] - kB.get("Sales", 0.0)
    units_delta = kA["Units"] - kB.get("Units", 0.0)
    aspA = kA["ASP"]
    aspB = kB.get("ASP", 0.0)
    asp_delta = aspA - aspB

    # Driver headline (top contributor by Sales_Δ)
    drv = drivers(dfA, dfB, driver_level)
    top_pos = drv[drv["Sales_Δ"] > 0].head(1)
    top_neg = drv[drv["Sales_Δ"] < 0].tail(1)  # bottom
    headline_bits = []
    if compare_mode != "None":
        headline_bits.append(f"Sales {('up' if sales_delta >= 0 else 'down')} **{money(abs(sales_delta))}** vs comparison.")
        headline_bits.append(f"Units {('up' if units_delta >= 0 else 'down')} **{abs(units_delta):,.0f}**.")
        if not np.isnan(asp_delta):
            headline_bits.append(f"ASP {('up' if asp_delta >= 0 else 'down')} **{money(abs(asp_delta))}**.")
        if not top_pos.empty:
            headline_bits.append(f"Top driver: **{top_pos.iloc[0][driver_level]}** ({money(float(top_pos.iloc[0]['Sales_Δ']))}).")
        if not top_neg.empty:
            headline_bits.append(f"Top drag: **{top_neg.iloc[0][driver_level]}** ({money(float(top_neg.iloc[0]['Sales_Δ']))}).")
    else:
        headline_bits.append("Choose a comparison mode to see drivers and deltas.")

    st.markdown(
        f"""
        <div style="border:1px solid rgba(0,0,0,0.08); background:linear-gradient(180deg, rgba(255,255,255,1), rgba(250,250,250,1));
                    border-radius:16px; padding:16px 18px; margin-bottom:14px;">
            <div style="font-size:13px; color:rgba(0,0,0,0.55); font-weight:700; letter-spacing:0.02em;">INTELLIGENCE SUMMARY</div>
            <div style="font-size:16px; margin-top:6px; line-height:1.4;">{" ".join(headline_bits)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1) KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kdelta(key):
        if compare_mode == "None":
            return ""
        cur = kA[key]
        prev = kB.get(key, 0.0)
        if key in ["Sales"]:
            return f"{pct_fmt(pct_change(cur, prev))} vs comp"
        if key in ["Units", "Active SKUs", "Active Retailers", "Active Vendors"]:
            return f"{pct_fmt(pct_change(cur, prev))} vs comp"
        if key == "ASP":
            return f"{pct_fmt(pct_change(cur, prev))} vs comp"
        return ""
    with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
    with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
    with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))
    with c4: kpi_card("Active SKUs", f"{kA['Active SKUs']:,}", kdelta("Active SKUs"))
    with c5: kpi_card("First Sales", f"{len(first_ever):,}" , "")
    with c6: kpi_card("New Placements", f"{len(placements):,}", "")

    st.write("")

    # 2) Drivers (two tables)
    st.subheader("Drivers (Contribution to change)")
    if compare_mode == "None":
        st.info("Select a comparison mode to compute drivers.")
    else:
        drv_show = drv.copy()
        drv_show = drv_show[(drv_show["Sales_A"] >= min_sales) | (drv_show["Sales_B"] >= min_sales)]
        pos = drv_show[drv_show["Sales_Δ"] > 0].head(10).copy()
        neg = drv_show[drv_show["Sales_Δ"] < 0].sort_values("Sales_Δ").head(10).copy()

        for d in (pos, neg):
            d["Sales_A"] = d["Sales_A"].map(money)
            d["Sales_B"] = d["Sales_B"].map(money)
            d["Sales_Δ"] = d["Sales_Δ"].map(lambda v: f"{money(v)}")
            d["Contribution_%"] = d["Contribution_%"].map(pct_fmt)

        left,right = st.columns(2)
        with left:
            st.markdown("**Top Positive Contributors**")
            render_df(pos[[driver_level,"Sales_A","Sales_B","Sales_Δ","Contribution_%"]], height=320)
        with right:
            st.markdown("**Top Negative Contributors**")
            render_df(neg[[driver_level,"Sales_A","Sales_B","Sales_Δ","Contribution_%"]], height=320)

    st.divider()

    # 3) Movers + Momentum
    st.subheader("Movers & Momentum")
    # Momentum at SKU level (within scope)
    mom = build_momentum(df_scope[df_scope["WeekEnd"] <= pA.end], "SKU", lookback_weeks=8)
    if not mom.empty:
        mom = mom[(mom["Sales (lookback)"] >= min_sales) | (mom["Units (lookback)"] >= min_units)].copy()
        top_inc = mom[mom["Trend"]=="Increasing"].head(10)
        top_dec = mom[mom["Trend"]=="Declining"].head(10)
        top_mom = mom.head(10)

        a,b,c = st.columns(3)
        with a:
            st.markdown("**Top Increasing (trend)**")
            render_df(top_inc[["SKU","Momentum","Momentum Label","Sales (lookback)"]], height=320)
        with b:
            st.markdown("**Top Declining (trend)**")
            render_df(top_dec[["SKU","Momentum","Momentum Label","Sales (lookback)"]], height=320)
        with c:
            st.markdown("**Top Momentum**")
            render_df(top_mom[["SKU","Momentum","Momentum Label","Sales (lookback)"]], height=320)
    else:
        st.info("Not enough history yet to compute momentum.")

    st.divider()

    # 4) Newness
    st.subheader("New Activity")
    a,b = st.columns(2)
    with a:
        st.markdown("**First Sale Ever (Launches)**")
        if first_ever.empty:
            st.caption("None in this period.")
        else:
            fe = first_ever.copy()
            fe["FirstWeek"] = fe["FirstWeek"].dt.date.astype(str)
            render_df(fe.rename(columns={"FirstWeek":"First Week"})[["SKU","First Week","FirstRetailer","FirstVendor"]], height=260)
    with b:
        st.markdown("**New Retailer Placements**")
        if placements.empty:
            st.caption("None in this period.")
        else:
            pl = placements.copy()
            pl["FirstWeek"] = pl["FirstWeek"].dt.date.astype(str)
            render_df(pl.rename(columns={"FirstWeek":"First Week"})[["SKU","Retailer","Vendor","First Week"]], height=260)

    st.divider()

    # 5) Weekly detail table
    st.subheader("Weekly Detail (Authoritative)")
    detail = dfA.copy()
    detail = detail[(detail["Sales"] >= min_sales) | (detail["Units"] >= min_units)].copy()
    show_cols = ["WeekEnd","Retailer","Vendor","SKU","Units","Price","Sales","SourceFile"]
    detail = detail[show_cols].sort_values(["WeekEnd","Retailer","Vendor","SKU"])
    detail["WeekEnd"] = detail["WeekEnd"].dt.date.astype(str)
    render_df(detail, height=460)

    st.divider()

    # -------------------------
    # Three advanced features
    # -------------------------
    st.header("Strategic Intelligence")

    # A) Contribution Tree
    st.subheader("1) Contribution Tree (Where did change come from?)")
    if compare_mode == "None":
        st.info("Select a comparison mode to use the contribution tree.")
    else:
        lvl1 = drivers(dfA, dfB, "Retailer")
        lvl1 = lvl1.sort_values("Sales_Δ", ascending=False)
        st.markdown("**Level 1 — Retailers**")
        render_df(lvl1[["Retailer","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].assign(
            Sales_A=lvl1["Sales_A"].map(money),
            Sales_B=lvl1["Sales_B"].map(money),
            Sales_Δ=lvl1["Sales_Δ"].map(money),
            Contribution_%=lvl1["Contribution_%"].map(pct_fmt),
        ), height=260)

        pick_r = st.selectbox("Drill into Retailer", options=["(none)"] + lvl1["Retailer"].tolist(), index=0)
        if pick_r != "(none)":
            dfA_r = dfA[dfA["Retailer"]==pick_r]
            dfB_r = dfB[dfB["Retailer"]==pick_r]
            lvl2 = drivers(dfA_r, dfB_r, "Vendor").sort_values("Sales_Δ", ascending=False)
            st.markdown(f"**Level 2 — Vendors inside {pick_r}**")
            render_df(lvl2[["Vendor","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].assign(
                Sales_A=lvl2["Sales_A"].map(money),
                Sales_B=lvl2["Sales_B"].map(money),
                Sales_Δ=lvl2["Sales_Δ"].map(money),
                Contribution_%=lvl2["Contribution_%"].map(pct_fmt),
            ), height=260)

            pick_v = st.selectbox("Drill into Vendor", options=["(none)"] + lvl2["Vendor"].tolist(), index=0)
            if pick_v != "(none)":
                dfA_v = dfA_r[dfA_r["Vendor"]==pick_v]
                dfB_v = dfB_r[dfB_r["Vendor"]==pick_v]
                lvl3 = drivers(dfA_v, dfB_v, "SKU").sort_values("Sales_Δ", ascending=False).head(30)
                st.markdown(f"**Level 3 — SKUs inside {pick_r} → {pick_v}**")
                render_df(lvl3[["SKU","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].assign(
                    Sales_A=lvl3["Sales_A"].map(money),
                    Sales_B=lvl3["Sales_B"].map(money),
                    Sales_Δ=lvl3["Sales_Δ"].map(money),
                    Contribution_%=lvl3["Contribution_%"].map(pct_fmt),
                ), height=360)

    st.divider()

    # B) SKU Lifecycle
    st.subheader("2) SKU Lifecycle (Launch → Growth → Mature → Decline → Dormant)")
    life_df_src = df_hist_for_new if show_full_history_lifecycle else df_scope
    life = lifecycle_table(life_df_src, pA, lookback_weeks=8)
    if life.empty:
        st.caption("Not enough data to compute lifecycle.")
    else:
        # small stage summary
        stage_counts = life["Stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage","Count"]
        left,right = st.columns([1,2])
        with left:
            st.markdown("**Stage Summary**")
            render_df(stage_counts, height=220)
        with right:
            st.markdown("**Top SKUs by stage (sorted by momentum)**")
            life_show = life.copy()
            life_show = life_show[life_show["Sales (lookback)"] >= min_sales].copy() if min_sales > 0 else life_show
            render_df(life_show.head(40), height=420)

    st.divider()

    # C) Opportunity Detector
    st.subheader("3) Opportunity Detector (Find expansion + gaps)")
    if compare_mode == "None":
        st.info("Select a comparison mode to power opportunity signals (needs a comparison).")
    else:
        opp = opportunity_detector(df_hist_for_new, dfA, dfB, pA)
        tabs = st.tabs(list(opp.keys()))
        for t, (name, odf) in zip(tabs, opp.items()):
            with t:
                if odf.empty:
                    st.caption("No signals found with current filters/thresholds.")
                else:
                    render_df(odf, height=420)

    # Footer
    st.caption("Tip: Use Scope + Driver Level together. Example: Scope=Retailer (Depot), Driver Level=Vendor to see which vendors drove Depot’s change.")
