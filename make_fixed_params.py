#!/usr/bin/env python3
"""
make_fixed_params.py
 Создает файл с фиксированными параметрами на основе календаря, добавляет лунные циклы для фана, но требует уточнения выходных/праздничный
 св соответствии с постановлениями правительства РФ
 Этот файл буде использоваться для подготовки полного датасета для обучения модели

Generates fixed_params.csv for dates 2024-01-01 .. 2025-12-31
with columns: date, dow, week_of_year, lunar_day, holiday_id, month, year, quarter

holiday_id:
  0 = normal weekday
  1 = weekend (Sat/Sun)
  2 = public holiday (overrides weekend flag)

Usage:
  python make_fixed_params.py --country RU --out fixed_params.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# optional imports
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except Exception:
    HOLIDAYS_AVAILABLE = False

try:
    from lunardate import LunarDate
    LUNARDATE_AVAILABLE = True
except Exception:
    LUNARDATE_AVAILABLE = False

from datetime import date, datetime, timedelta

def approximate_lunar_day(dt: date, ref=date(2000,1,6)):
    """
    Approximate lunar day using 29.530588853-day synodic month (simple modular arithmetic).
    Returns 1..30 (rounded).
    """
    days = (dt - ref).days
    synodic = 29.530588853
    lunar = int((days % synodic) + 1)
    if lunar <= 0:
        lunar = 1
    if lunar > 30:
        lunar = lunar % 30
        if lunar == 0:
            lunar = 30
    return lunar

def lunar_day_for_date(dt: date):
    """
    Prefer lunardate library if available (gives lunar year/month/day),
    otherwise fallback to approximate cycle.
    Returns int 1..30
    """
    if LUNARDATE_AVAILABLE:
        try:
            ld = LunarDate.fromSolarDate(dt.year, dt.month, dt.day)
            # LunarDate gives lunar month/day; return day
            return int(ld.day)
        except Exception:
            return approximate_lunar_day(dt)
    else:
        return approximate_lunar_day(dt)

def build_fixed_params(start="2024-01-01", end="2025-12-31", country="BE", prov=None, out_path="fixed_params.csv"):
    # build date range
    dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday  # 0=Mon .. 6=Sun
    # ISO week of year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    # month/year/quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # lunar day (1..30) using lunardate if available, else approximate
    df["lunar_day"] = df["date"].dt.date.apply(lunar_day_for_date).astype(int)

    # holiday id using python-holidays if available
    if HOLIDAYS_AVAILABLE:
        if prov:
            cal = holidays.CountryHoliday(country, prov=prov)
        else:
            cal = holidays.CountryHoliday(country)
        def holiday_id_for_day(d: date):
            # d is a datetime.date
            if d in cal:
                return 2  # public holiday
            if d.weekday() >= 5:
                return 1  # weekend
            return 0
    else:
        # fallback: mark weekends as 1, weekdays 0
        def holiday_id_for_day(d: date):
            return 1 if d.weekday() >= 5 else 0

    df["holiday_id"] = df["date"].dt.date.apply(holiday_id_for_day).astype(int)

    # If you'd like separate columns for is_weekend/is_holiday flags:
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_public_holiday"] = (df["holiday_id"] == 2).astype(int)

    # Save CSV
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"Saved fixed params to {out.resolve()} (rows: {len(df)})")
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--country", type=str, default="BE", help="Country code for holidays (e.g. 'BE','RU','US')")
    p.add_argument("--prov", type=str, default=None, help="Subdivision/province code (optional)")
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--out", type=str, default="fixed_params.csv")
    args = p.parse_args()

    if not HOLIDAYS_AVAILABLE:
        print("[WARN] python-holidays not installed. Install with 'pip install holidays' to get public holidays. Falling back to weekend-only holiday_id.")
    if not LUNARDATE_AVAILABLE:
        print("[INFO] lunardate not installed. Using approximate lunar-day cycle. Install with 'pip install lunardate' for better lunar days.")

    build_fixed_params(start=args.start, end=args.end, country=args.country, prov=args.prov, out_path=args.out)
