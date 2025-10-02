#!/usr/bin/env python3
"""
generate_many_itps_with_types.py

Генерация синтетических почасовых данных потребления холодной воды
за период 2024-01-01 -> 2025-12-31 для множества ITP (itp_uuid, itp_id).
Добавлены аномалии и пробелы

Добавлено: каждый ITP имеет consumer_type из набора:
  - "residential" (жилые дома)
  - "industrial" (промышленные предприятия)
  - "school" (школы)
  - "kindergarten" (детские сады)
  - "mall" (торговые центры)

В выходе: колонки ts, itp_id, itp_uuid, consumer_type, consumer_type_id, cold_flow_m3_per_hour, ...
Метаданные содержат параметры генерации для каждого ITP.

Запуск генерации датасета (в самой нижней части настройки, для скорости сделано для 20 ИТП в 1 файл parquet):
python generate_many_itps_with_types.py
"""

import os
import argparse
import math
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
from tqdm import tqdm

# ---------- Consumer types and probabilities ----------
CONSUMER_TYPES = [
    {"id": 0, "key": "residential", "ru": "жилые дома"},
    {"id": 1, "key": "industrial", "ru": "промышленные предприятия"},
    {"id": 2, "key": "school", "ru": "школы"},
    {"id": 3, "key": "kindergarten", "ru": "детские сады"},
    {"id": 4, "key": "mall", "ru": "торговые центры"},
    {"id": 5, "key": "others", "ru": "другое"},
]
# по умолчанию вероятности типов (сумма = 1.0)
DEFAULT_TYPE_PROBS = {
    "residential": 0.70,
    "industrial": 0.05,
    "school": 0.05,
    "kindergarten": 0.05,
    "mall": 0.10,
    "others": 0.05,
}

# ---------- Utility generators (based on previous single-itp code) ----------
def generate_base_profile_for_index(idx, morning_peak, evening_peak, base_level, weekend_factor, seasonal_amp, seed_local=0):
    """
    Возвращает numpy array базового (детерминированного) профиля длины idx
    idx: pandas.DatetimeIndex с частотой H
    Коэффициенты — скаляры.
    """
    hours = idx.hour.values
    morning = morning_peak * np.exp(-0.5 * ((hours - 8) / 2.0) ** 2)
    evening = evening_peak * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2)
    diurnal = base_level + morning + evening

    dow = idx.dayofweek.values
    is_weekend = (dow >= 5).astype(float)
    weekly = 1.0 - (1.0 - weekend_factor) * is_weekend

    dayofyear = idx.dayofyear.values
    seasonal = 1.0 + seasonal_amp * np.sin(2 * np.pi * (dayofyear - 200) / 365.25)

    return diurnal * weekly * seasonal

def add_noise_and_basic_randomness(base, rng, relative_noise_std, occasional_zero_prob, trend_scale):
    noise = rng.normal(loc=0.0, scale=relative_noise_std * base.mean(), size=base.shape)
    trend = np.linspace(0, trend_scale * base.mean(), len(base))
    result = base + noise + trend
    zero_mask = rng.random(len(base)) < occasional_zero_prob
    result[zero_mask] = 0.0
    result = np.clip(result, a_min=0.0, a_max=None)
    return result, zero_mask

def inject_anomalies_bulk(values, idx, rng, leak_days_prob, leak_mult_range, spike_prob, outage_prob):
    n = len(values)
    leak_mask = np.zeros(n, dtype=bool)
    spike_mask = np.zeros(n, dtype=bool)
    outage_mask = np.zeros(n, dtype=bool)

    unique_dates = np.unique(idx.date)
    for d in unique_dates:
        if rng.random() < leak_days_prob:
            day_idx = np.where(idx.normalize() == pd.Timestamp(d))[0]
            if len(day_idx) == 0:
                continue
            start = int(day_idx[0] + rng.integers(0, 24))
            max_days = rng.integers(1, 6)  # 1..5 days
            dur = int(rng.integers(6, 24 * max_days))
            end = min(n, start + dur)
            mult = float(rng.uniform(leak_mult_range[0], leak_mult_range[1]))
            values[start:end] = values[start:end] * mult
            leak_mask[start:end] = True

    spike_positions = rng.random(n) < spike_prob
    if spike_positions.any():
        amps = rng.uniform(2.0, 6.0, size=spike_positions.sum())
        values[spike_positions] = values[spike_positions] * amps + rng.normal(0, 0.1, size=spike_positions.sum())
        spike_mask[spike_positions] = True

    outage_positions = rng.random(n) < outage_prob
    for i in np.where(outage_positions)[0]:
        dur = int(rng.integers(1, 6))
        end = min(n, i + dur)
        values[i:end] = 0.0
        outage_mask[i:end] = True

    values = np.clip(values, a_min=0.0, a_max=None)
    return values, leak_mask, spike_mask, outage_mask

# ---------- Parameter sampling by consumer type ----------
def sample_itp_params_by_type(rng, consumer_type_key):
    """
    Возвращает словарь параметров, уникальный для ITP, в зависимости от consumer_type_key.
    Диапазоны настроены для разных типов, чтобы отражать реальные отличия.
    """
    if consumer_type_key == "residential":
        # жилые дома: выраженные утренние/вечерние пики, умеренная база
        params = {
            "base_level": float(rng.uniform(0.3, 1.2)),
            "morning_peak": float(rng.uniform(1.0, 2.0)),
            "evening_peak": float(rng.uniform(1.2, 2.6)),
            "weekend_factor": float(rng.uniform(0.7, 0.95)),
            "seasonal_amp": float(rng.uniform(0.05, 0.35)),
            "relative_noise_std": float(rng.uniform(0.05, 0.12)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0008)),
            "trend_scale": float(rng.uniform(-0.01, 0.02)),
            "leak_days_prob": float(rng.uniform(0.0003, 0.0025)),
            "leak_mult_range": (float(rng.uniform(1.4, 1.8)), float(rng.uniform(1.8, 3.0))),
            "spike_prob": float(rng.uniform(0.0008, 0.004)),
            "outage_prob": float(rng.uniform(0.00005, 0.001)),
        }
    elif consumer_type_key == "industrial":
        # промышленность: высокий базовый расход, менее выраженные суточные пики,
        # более частые большие аномалии (т.к. технологические колебания)
        params = {
            "base_level": float(rng.uniform(2.0, 10.0)),
            "morning_peak": float(rng.uniform(0.3, 1.0)),
            "evening_peak": float(rng.uniform(0.3, 1.0)),
            "weekend_factor": float(rng.uniform(0.6, 1.0)),  # у некоторых 24/7, у некоторых нет
            "seasonal_amp": float(rng.uniform(0.0, 0.2)),
            "relative_noise_std": float(rng.uniform(0.05, 0.2)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0002)),
            "trend_scale": float(rng.uniform(-0.02, 0.05)),
            "leak_days_prob": float(rng.uniform(0.0005, 0.0035)),
            "leak_mult_range": (float(rng.uniform(1.8, 2.5)), float(rng.uniform(2.5, 4.0))),
            "spike_prob": float(rng.uniform(0.001, 0.006)),
            "outage_prob": float(rng.uniform(0.00001, 0.0008)),
        }
    elif consumer_type_key == "school":
        # школы: выраженный дневной пик в рабочие дни, низкое потребление в вечер/выходные
        params = {
            "base_level": float(rng.uniform(0.5, 2.0)),
            "morning_peak": float(rng.uniform(1.5, 3.0)),
            "evening_peak": float(rng.uniform(0.2, 0.8)),
            "weekend_factor": float(rng.uniform(0.3, 0.6)),
            "seasonal_amp": float(rng.uniform(0.05, 0.25)),
            "relative_noise_std": float(rng.uniform(0.04, 0.12)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0009)),
            "trend_scale": float(rng.uniform(-0.01, 0.015)),
            "leak_days_prob": float(rng.uniform(0.0002, 0.0015)),
            "leak_mult_range": (float(rng.uniform(1.4, 1.9)), float(rng.uniform(1.9, 3.0))),
            "spike_prob": float(rng.uniform(0.0005, 0.003)),
            "outage_prob": float(rng.uniform(0.00005, 0.0008)),
        }
    elif consumer_type_key == "kindergarten":
        # детсады: утренний пик и ранний дневной, более регулярное потребление в будни
        params = {
            "base_level": float(rng.uniform(0.4, 1.5)),
            "morning_peak": float(rng.uniform(1.6, 3.2)),
            "evening_peak": float(rng.uniform(0.3, 1.0)),
            "weekend_factor": float(rng.uniform(0.2, 0.5)),
            "seasonal_amp": float(rng.uniform(0.04, 0.2)),
            "relative_noise_std": float(rng.uniform(0.04, 0.1)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0009)),
            "trend_scale": float(rng.uniform(-0.01, 0.02)),
            "leak_days_prob": float(rng.uniform(0.0002, 0.0018)),
            "leak_mult_range": (float(rng.uniform(1.4, 1.8)), float(rng.uniform(1.8, 2.8))),
            "spike_prob": float(rng.uniform(0.0006, 0.0035)),
            "outage_prob": float(rng.uniform(0.00005, 0.0008)),
        }
    elif consumer_type_key == "mall":
        # торговые центры: высокий базовый + дневной пик, более выраженный в выходные
        params = {
            "base_level": float(rng.uniform(1.0, 5.0)),
            "morning_peak": float(rng.uniform(0.8, 1.6)),
            "evening_peak": float(rng.uniform(1.0, 2.4)),
            "weekend_factor": float(rng.uniform(0.9, 1.4)),  # в выходные нагрузка может расти
            "seasonal_amp": float(rng.uniform(0.05, 0.3)),
            "relative_noise_std": float(rng.uniform(0.06, 0.15)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0004)),
            "trend_scale": float(rng.uniform(-0.01, 0.03)),
            "leak_days_prob": float(rng.uniform(0.0004, 0.0028)),
            "leak_mult_range": (float(rng.uniform(1.6, 2.2)), float(rng.uniform(2.0, 3.2))),
            "spike_prob": float(rng.uniform(0.0008, 0.0045)),
            "outage_prob": float(rng.uniform(0.00003, 0.0009)),
        }
    elif consumer_type_key == "others":
        # торговые центры: высокий базовый + дневной пик, более выраженный в выходные
        params = {
            "base_level": float(rng.uniform(1.0, 4.0)),
            "morning_peak": float(rng.uniform(0.8, 1.6)),
            "evening_peak": float(rng.uniform(1.0, 2.0)),
            "weekend_factor": float(rng.uniform(0.9, 1.4)),  # в выходные нагрузка может расти
            "seasonal_amp": float(rng.uniform(0.05, 0.2)),
            "relative_noise_std": float(rng.uniform(0.06, 0.15)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0004)),
            "trend_scale": float(rng.uniform(-0.01, 0.03)),
            "leak_days_prob": float(rng.uniform(0.0004, 0.0028)),
            "leak_mult_range": (float(rng.uniform(1.6, 2.2)), float(rng.uniform(2.0, 3.2))),
            "spike_prob": float(rng.uniform(0.0010, 0.0065)),
            "outage_prob": float(rng.uniform(0.00003, 0.0009)),
        }
    else:
        # fallback: residential-like
        params = {
            "base_level": float(rng.uniform(0.3, 1.2)),
            "morning_peak": float(rng.uniform(1.0, 2.0)),
            "evening_peak": float(rng.uniform(1.2, 2.6)),
            "weekend_factor": float(rng.uniform(0.7, 0.95)),
            "seasonal_amp": float(rng.uniform(0.05, 0.35)),
            "relative_noise_std": float(rng.uniform(0.05, 0.12)),
            "occasional_zero_prob": float(rng.uniform(0.0, 0.0008)),
            "trend_scale": float(rng.uniform(-0.01, 0.02)),
            "leak_days_prob": float(rng.uniform(0.0003, 0.0025)),
            "leak_mult_range": (float(rng.uniform(1.4, 1.8)), float(rng.uniform(1.8, 3.0))),
            "spike_prob": float(rng.uniform(0.0008, 0.004)),
            "outage_prob": float(rng.uniform(0.00005, 0.001)),
        }
    # ensure proper ordering in leak_mult_range
    a, b = params["leak_mult_range"]
    if a > b:
        params["leak_mult_range"] = (b, a)
    return params

# ---------- Main ITP generator for one ITP ----------
def generate_single_itp_dataframe(idx, itp_id, itp_uuid, consumer_type_key, params, rng):
    """
    Возвращает pandas.DataFrame для одного ITP: колонки ts, itp_id, consumer_type, consumer_type_id, cold_flow_m3_per_hour,
    is_zero_generated, is_leak, is_spike, is_outage, plus parameters as meta columns.
    """
    base = generate_base_profile_for_index(
        idx,
        morning_peak=params["morning_peak"],
        evening_peak=params["evening_peak"],
        base_level=params["base_level"],
        weekend_factor=params["weekend_factor"],
        seasonal_amp=params["seasonal_amp"],
    )
    vals, zero_mask = add_noise_and_basic_randomness(
        base, rng,
        relative_noise_std=params["relative_noise_std"],
        occasional_zero_prob=params["occasional_zero_prob"],
        trend_scale=params["trend_scale"]
    )
    vals, leak_mask, spike_mask, outage_mask = inject_anomalies_bulk(
        vals, idx, rng,
        leak_days_prob=params["leak_days_prob"],
        leak_mult_range=params["leak_mult_range"],
        spike_prob=params["spike_prob"],
        outage_prob=params["outage_prob"]
    )

    # map consumer_type_key to id
    type_id = next((t["id"] for t in CONSUMER_TYPES if t["key"] == consumer_type_key), None)

    df = pd.DataFrame({
        "date": idx,
        "itp_id": str(itp_id),
        "itp_uuid": str(itp_uuid),
        "consumer_type": consumer_type_key,
        "consumer_type_id": int(type_id) if type_id is not None else None,
        "cold_flow_m3_per_hour": vals,
        "is_zero_generated": zero_mask.astype(int),
        "is_leak": leak_mask.astype(int),
        "is_spike": spike_mask.astype(int),
        "is_outage": outage_mask.astype(int),
    })
    # add scalar params as columns prefixed with param_
    for k, v in params.items():
        if np.isscalar(v):
            df[f"param_{k}"] = v
    return df

# ---------- Batch generation and writing ----------
def choose_consumer_type(master_rng, type_probs=None):
    """
    Выбирает consumer_type_key по вероятностям type_probs (dict). Возвращает ключ типа.
    """
    if type_probs is None:
        type_probs = DEFAULT_TYPE_PROBS
    keys = list(type_probs.keys())
    probs = np.array([type_probs[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    choice = master_rng.choice(keys, p=probs)
    return choice

def generate_many_itps(num_itps=1000, batch_size=100, out_dir="out_data",
                       fmt="parquet", seed=123, tz=None, type_probs=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2024-01-01 00:00", tz=tz)
    end = pd.Timestamp("2025-12-31 23:00", tz=tz)
    idx = pd.date_range(start=start, end=end, freq="h", tz=tz)

    metadata = []
    created_files = []
    master_rng = np.random.default_rng(seed)

    total_batches = math.ceil(num_itps / batch_size)
    itp_counter = 0

    print(f"Generating {num_itps} ITPs in {total_batches} batches (batch_size={batch_size})")
    for batch_idx in range(total_batches):
        batch_count = min(batch_size, num_itps - itp_counter)
        frames = []
        for i in range(batch_count):
            itp_id = i            
            itp_uuid = uuid.uuid4()
            itp_seed = master_rng.integers(0, 2**31 - 1)
            rng = np.random.default_rng(int(itp_seed))

            consumer_type_key = choose_consumer_type(master_rng, type_probs=type_probs)
            params = sample_itp_params_by_type(rng, consumer_type_key)
            df_itp = generate_single_itp_dataframe(idx,itp_id, itp_uuid, consumer_type_key, params, rng)
            frames.append(df_itp)

            md = {"itp_id": itp_id,"itp_uuid": str(itp_uuid), "seed": int(itp_seed), "consumer_type": consumer_type_key}
            md.update(params)
            metadata.append(md)

            itp_counter += 1

        batch_df = pd.concat(frames, axis=0, ignore_index=True)

        out_name = out_dir / f"itps_batch_{batch_idx:05d}.{ 'parquet' if fmt=='parquet' else 'csv.gz'}"
        if fmt == "parquet":
            try:
                batch_df.to_parquet(out_name, index=False)
            except Exception as e:
                print("Ошибка записи parquet:", e, file=sys.stderr)
                print("Попытка записать в csv.gz")
                alt_name = str(out_name).replace(".parquet", ".csv.gz")
                batch_df.to_csv(alt_name, index=False, compression="gzip")
                created_files.append(alt_name)
            else:
                created_files.append(str(out_name))
        else:
            batch_df.to_csv(out_name, index=False, compression="gzip")
            created_files.append(str(out_name))

        del batch_df
        del frames
        print(f"Wrote batch {batch_idx+1}/{total_batches} -> {created_files[-1]} (ITP count so far: {itp_counter})")

    meta_path = out_dir / "itp_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # also save consumer types mapping
    types_map = {t["key"]: {"id": t["id"], "ru": t["ru"]} for t in CONSUMER_TYPES}
    with open(out_dir / "consumer_types.json", "w", encoding="utf-8") as f:
        json.dump(types_map, f, indent=2, ensure_ascii=False)

    print("Done. Files created:", len(created_files))
    return {"files": created_files, "metadata": str(meta_path), "consumer_types": str(out_dir / "consumer_types.json")}

# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic hourly cold-flow data for many ITPs with consumer types")
    parser.add_argument("--num-itps", type=int, default=10000, help="Number of ITPs to generate")
    parser.add_argument("--batch-size", type=int, default=10000, help="ITPs per output file (memory friendly)")
    parser.add_argument("--out-dir", type=str, default="./synthetic_itps_out", help="Output directory")
    parser.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet", help="Output format")
    parser.add_argument("--seed", type=int, default=123, help="Global random seed")
    parser.add_argument("--tz", type=str, default=None, help="Timezone for timestamps, e.g. 'Europe/Riga' or empty")
    return parser.parse_args()

if __name__ == "__main__":
    #args = parse_args()
    # you can modify DEFAULT_TYPE_PROBS here if desired
    res = generate_many_itps(num_itps=20,
                             batch_size=20,
                             out_dir="./synthetic_itps_out",
                             fmt='parquet',
                             seed=42,
                             tz=None,
                             type_probs=DEFAULT_TYPE_PROBS)
    
                           # (num_itps=args.num_itps,
                           #  batch_size=args.batch_size,
                           #  out_dir=args.out_dir,
                           #  fmt=args.format,
                           #  seed=args.seed,
                           #  tz=args.tz,
                           #  type_probs=DEFAULT_TYPE_PROBS)
    print("Result:", res)
