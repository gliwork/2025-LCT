#!/usr/bin/env python3
"""
train_transformer_holiday.py

Модуль обучения модели из файла forecast_with_holiday_transformer.py
Забирает данные из подготовленного датасета расходов для множества ИТП
и отправляет на обучение. Может работать медленно с большим датасетом,
пожтому датасет можно готовить заранее, записать на диск и загружать при помощи модуля dataset_loader.py
На выходе - обученная модель в папке './chkpts'
Скрипт для FullForecastModelTransformerHoliday (Transformer-based holiday encoder),
со встроенной нормализацией для параметров hist_matrix и avg_daily_temps_m.
"""

import os
import argparse
import random
import json
from pathlib import Path
import time

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import parquet_to_batches

# -------------------------------
# Импорт модели (предполагается подготовленная модель)
# -------------------------------
MODEL_MODULE_NAME = "forecast_with_holiday_transformer"  # ваш файл с моделью
try:
    mod = __import__(MODEL_MODULE_NAME, fromlist=["FullForecastModelTransformerHoliday", "Config"])
    FullForecastModel = mod.FullForecastModelTransformerHoliday
    Config = mod.Config
except Exception as e:
    raise RuntimeError("Поместите файл 'forecast_with_holiday_transformer.py' в папку скрипта") from e

# -------------------------------
# EarlyStopping helper
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-5, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.save_path = save_path

    def step(self, metric, model_state=None):
        improved = (self.best - metric) > self.min_delta
        if improved:
            self.best = metric
            self.wait = 0
            if self.save_path and model_state is not None:
                torch.save(model_state, self.save_path)
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience

# -------------------------------
# Synthetic Dataset
# -------------------------------
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, N=2000):
        self.cfg = cfg
        self.N = N
        m = cfg.m
        n = cfg.n
        self.hist = np.random.rand(N, 1, 24, m).astype(np.float32)
        self.hour = (np.random.randint(0, 24, size=(N,1)).astype(np.float32) / 23.0)
        self.dow = (np.random.randint(0, 7, size=(N,1)).astype(np.float32) / 6.0)
        self.week = (np.random.randint(1, 53, size=(N,1)).astype(np.float32) / 52.0)
        self.lunar = (np.random.randint(1, 30, size=(N,1)).astype(np.float32) / 29.0)
        self.temps = np.random.randn(N, m).astype(np.float32)
        self.holidays = np.random.randint(0, cfg.n_holiday_ids, size=(N, m + n)).astype(np.int64)
        self.itp_id = np.random.randint(0, cfg.n_itp, size=(N,))
        #self.consumer_id = np.random.randint(0, cfg.n_consumer_types, size=(N,))
        #self.seg_id = np.random.randint(0, cfg.n_network_segments, size=(N,))
        self.target = np.random.rand(N, n * 24).astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            "hist_matrix": self.hist[idx],
            "hour_start": self.hour[idx],
            "dow": self.dow[idx],
            "week_of_year": self.week[idx],
            "lunar_day": self.lunar[idx],
            "avg_daily_temps_m": self.temps[idx],
            "holiday_seq_mn": self.holidays[idx],
            "itp_id": np.int64(self.itp_id[idx]),
           # "consumer_type_id": np.int64(self.consumer_id[idx]),
           # "network_segment_id": np.int64(self.seg_id[idx]),
            "target": self.target[idx],
        }

#Prepare dataset
def prepare_dataset(parquet_files, fixed_params_csv, m=14, n=3):
    """
    Args:
        parquet_files: list of paths к parquet файлам с часовой историей water consumption
        fixed_params_csv: path к csv с daily params: date,lunar_day,dow,week_of_year,holiday_id
        m: кол-во прошлых дней для hist_matrix
        n: кол-во дней для прогноза
    Returns:
        dataset: list of dict, каждый dict - одна запись для модели
    """
    # 1. читаем fixed_params.csv
    fixed_df = pd.read_csv(fixed_params_csv, parse_dates=["date"])
    fixed_df.set_index("date", inplace=True)

    dataset = []

    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)  # должен содержать columns: ['date','itp_id','consumption']
        df['date'] = df['date'].dt.floor('D')  # дата без времени
        df['hour'] = df['date'].dt.hour

        # группируем по ИТП
        for itp_id, grp in tqdm(df.groupby('itp_id'), desc=f"Processing {pq_file}"):
            grp = grp.sort_values('date')
            dates = grp['date'].unique()

            for i in range(m, len(dates)-n):
                # предыдущие m дней
                prev_dates = dates[i-m:i]
                future_dates = dates[i:i+n]

                # hist_matrix: 24 x m
                hist_matrix = []
                avg_daily_temps_m = []  # placeholder если есть temp per day
                for d in prev_dates:
                    day_data = grp[grp['date']==d].sort_values('hour')['cold_flow_m3_per_hour'].values
                    if len(day_data) < 24:
                        day_data = np.pad(day_data, (0,24-len(day_data)), 'edge')
                    hist_matrix.append(day_data)
                    # temp placeholder, здесь можно вставить реальные avg temp
                    avg_daily_temps_m.append(day_data.mean())

                hist_matrix = np.stack(hist_matrix, axis=1)  # [24, m]
                avg_daily_temps_m = np.array(avg_daily_temps_m)

                # holiday_seq_mn: m+n
                holiday_seq_mn = []
                for d in list(prev_dates) + list(future_dates):
                    if d in fixed_df.index:
                        holiday_seq_mn.append(fixed_df.loc[d, 'holiday_id'])
                    else:
                        holiday_seq_mn.append(0)  # default

                # дополнительные признаки
                hour_start = np.array([0.0], dtype=np.float32)  # пример
                dow = np.array([fixed_df.loc[future_dates[0], 'dow']], dtype=np.float32)
                week_of_year = np.array([fixed_df.loc[future_dates[0], 'week_of_year']], dtype=np.float32)
                lunar_day = np.array([fixed_df.loc[future_dates[0], 'lunar_day']], dtype=np.float32)

                target = []
                for d in future_dates:
                    day_data = grp[grp['date']==d].sort_values('hour')['cold_flow_m3_per_hour'].values
                    if len(day_data) < 24:
                        day_data = np.pad(day_data, (0,24-len(day_data)), 'edge')
                    target.append(day_data)
                target = np.concatenate(target)  # n*24

                dataset.append({
                    "itp_id": int(itp_id),
                    "hist_matrix": hist_matrix.astype(np.float32)[np.newaxis,:,:],  # [1,24,m]
                    "avg_daily_temps_m": avg_daily_temps_m.astype(np.float32),
                    "holiday_seq_mn": np.array(holiday_seq_mn, dtype=np.int64),
                    "hour_start": hour_start,
                    "dow": dow,
                    "week_of_year": week_of_year,
                    "lunar_day": lunar_day,
                    "target": target.astype(np.float32)
                })
    return dataset

#Collate - fast
def collate_fn(batch):
    # Collect lists
    print(type(batch))
    hist_matrix = np.array([sample["hist_matrix"] for sample in batch], dtype=np.float32)
    avg_daily_temps_m = np.array([sample["avg_daily_temps_m"] for sample in batch], dtype=np.float32)
    holiday_seq_mn = np.array([sample["holiday_seq_mn"] for sample in batch], dtype=np.int64)
    target = np.array([sample["target"] for sample in batch], dtype=np.float32)

    hour_start = np.array([sample["hour_start"] for sample in batch], dtype=np.float32)
    dow = np.array([sample["dow"] for sample in batch], dtype=np.float32)
    week_of_year = np.array([sample["week_of_year"] for sample in batch], dtype=np.float32)
    lunar_day = np.array([sample["lunar_day"] for sample in batch], dtype=np.float32)

    itp_id = np.array([sample["itp_id"] for sample in batch], dtype=np.int64)
    consumer_type_id = np.array([sample["consumer_type_id"] for sample in batch], dtype=np.int64)
    network_segment_id = np.array([sample["network_segment_id"] for sample in batch], dtype=np.int64)

    # Convert to tensors in one go
    batch_dict = {
        "hist_matrix": torch.from_numpy(hist_matrix),         # [B,24,m]
        "avg_daily_temps_m": torch.from_numpy(avg_daily_temps_m), # [B,m]
        "holiday_seq_mn": torch.from_numpy(holiday_seq_mn),   # [B,m+n]
        "hour_start": torch.from_numpy(hour_start),
        "dow": torch.from_numpy(dow),
        "week_of_year": torch.from_numpy(week_of_year),
        "lunar_day": torch.from_numpy(lunar_day),
        "itp_id": torch.from_numpy(itp_id),
        "consumer_type_id": torch.from_numpy(consumer_type_id),
        "network_segment_id": torch.from_numpy(network_segment_id),
        "target": torch.from_numpy(target),                  # [B,n*24]
    }
    return batch_dict

# -------------------------------
# Collate function с нормализацией
# -------------------------------
class CollateWithNormalization:
    def __init__(self, normalizers_path):
        normalizers_path = Path(normalizers_path)
        if not normalizers_path.exists():
            raise FileNotFoundError(f"Normalizers file not found: {normalizers_path}")
        with open(normalizers_path, "r") as f:
            self.norm = json.load(f)

        # store mean/std as torch scalars for normalization
        self.hist_mean = torch.tensor(self.norm["hist_matrix"]["mean"], dtype=torch.float32)
        self.hist_std = torch.tensor(self.norm["hist_matrix"]["std"], dtype=torch.float32)
        self.temp_mean = torch.tensor(self.norm["avg_daily_temps_m"]["mean"], dtype=torch.float32)
        self.temp_std = torch.tensor(self.norm["avg_daily_temps_m"]["std"], dtype=torch.float32)

    def __call__(self, batch):
        # stack numpy arrays first -> then torch.from_numpy
        hist_np = np.array([b["hist_matrix"] for b in batch], dtype=np.float32)
        temps_np = np.array([b["avg_daily_temps_m"] for b in batch], dtype=np.float32)
        holiday_np = np.array([b["holiday_seq_mn"] for b in batch], dtype=np.int64)
        target_np = np.array([b["target"] for b in batch], dtype=np.float32)

        hour_np = np.array([b["hour_start"] for b in batch], dtype=np.float32)
        dow_np = np.array([b["dow"] for b in batch], dtype=np.float32)
        week_np = np.array([b["week_of_year"] for b in batch], dtype=np.float32)
        lunar_np = np.array([b["lunar_day"] for b in batch], dtype=np.float32)
        itp_np = np.array([b["itp_id"] for b in batch], dtype=np.int64)

        # convert to tensors
        hist = torch.from_numpy(hist_np)
        avg_temps = torch.from_numpy(temps_np)
        holiday_seq_mn = torch.from_numpy(holiday_np)
        target = torch.from_numpy(target_np)

        hour_start = torch.from_numpy(hour_np)
        dow = torch.from_numpy(dow_np)
        week_of_year = torch.from_numpy(week_np)
        lunar_day = torch.from_numpy(lunar_np)
        itp_id = torch.from_numpy(itp_np)

        # normalize
        hist = (hist - self.hist_mean) / (self.hist_std + 1e-8)
        avg_temps = (avg_temps - self.temp_mean) / (self.temp_std + 1e-8)

        return {
            "hist_matrix": hist,
            "hour_start": hour_start,
            "dow": dow,
            "week_of_year": week_of_year,
            "lunar_day": lunar_day,
            "avg_daily_temps_m": avg_temps,
            "holiday_seq_mn": holiday_seq_mn,
            "itp_id": itp_id,
            "target": target,
        }

# -------------------------------
# Функция вычисления нормализаторов
# -------------------------------
def compute_normalizers(dataloader, device="cpu", out_path="normalizers.json"):
    sum_hist = 0.0
    sumsq_hist = 0.0
    count_hist = 0
    sum_temp = 0.0
    sumsq_temp = 0.0
    count_temp = 0

    for batch in dataloader:
        hist = batch["hist_matrix"].to(device)
        temps = batch["avg_daily_temps_m"].to(device)
        hist_flat = hist.view(-1)
        temps_flat = temps.view(-1)
        sum_hist += hist_flat.sum().item()
        sumsq_hist += (hist_flat ** 2).sum().item()
        count_hist += hist_flat.numel()
        sum_temp += temps_flat.sum().item()
        sumsq_temp += (temps_flat ** 2).sum().item()
        count_temp += temps_flat.numel()

    mean_hist = sum_hist / count_hist
    std_hist = (sumsq_hist / count_hist - mean_hist ** 2) ** 0.5
    mean_temp = sum_temp / count_temp
    std_temp = (sumsq_temp / count_temp - mean_temp ** 2) ** 0.5

    normalizers = {
        "hist_matrix": {"mean": mean_hist, "std": std_hist},
        "avg_daily_temps_m": {"mean": mean_temp, "std": std_temp},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(normalizers, f, indent=2)
    print(f"[INFO] Normalizers saved to {out_path}")
    return normalizers

# -------------------------------
# Training
# -------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#здесь - подготовка датасета и батчей и даталоадера

def prepare_dataloaders(args, cfg):
    # demo synthetic dataset
    #SyntheticDataset(cfg, N=2000)
# Здесь синтетика, но можно заменить на норм
    parquet_files = ["./synthetic_itps_out/itps_batch_00000.parquet"]              #address of parquet file
    fixed_params_csv = "fixed_params.csv"
    ds_total = prepare_dataset(parquet_files, fixed_params_csv, m=14, n=3)
    					
    train_N = int(0.8 * len(ds_total))
    val_N = len(ds_total) - train_N
    train_ds, val_ds = random_split(ds_total, [train_N, val_N])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader

def train(args):
    set_seed(args.seed)
    cfg = Config()
    cfg.m = args.m
    cfg.n = args.n
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.epochs = args.epochs

    train_loader, val_loader = prepare_dataloaders(args, cfg)

    # -------------------------------
    # нормализаторы
    # -------------------------------
    normalizers_path = Path(args.out_dir) / "normalizers.json"
    compute_normalizers(train_loader, device=args.device, out_path=normalizers_path)
    collate_fn_with_norm = CollateWithNormalization(normalizers_path)

    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_with_norm, num_workers=args.num_workers)
    val_loader = DataLoader(val_loader.dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_with_norm, num_workers=args.num_workers)

    # -------------------------------
    # модель
    # -------------------------------
    model = FullForecastModel(cfg).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)#, verbose=True
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pt"
    last_ckpt = out_dir / "last_model.pt"
    early_stopper = EarlyStopping(patience=args.early_stop_patience, min_delta=1e-6, save_path=str(best_ckpt))

    # optional tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        use_tb = True
    except Exception:
        writer = None
        use_tb = False

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_count = 0
        t0 = time.time()
        for batch in train_loader:
            
            optimizer.zero_grad()
            preds = model(batch["hist_matrix"].to(args.device),
                          batch["hour_start"].to(args.device),
                          batch["dow"].to(args.device),
                          batch["week_of_year"].to(args.device),
                          batch["lunar_day"].to(args.device),
                          batch["avg_daily_temps_m"].to(args.device),
                          batch["holiday_seq_mn"].to(args.device),
                          batch["itp_id"].to(args.device),
                          #batch["consumer_type_id"].to(args.device),
                         # batch["network_segment_id"].to(args.device)
)
            loss = loss_fn(preds, batch["target"].to(args.device))
            loss.backward()
            optimizer.step()

            bs = preds.size(0)
            train_loss_acc += loss.item() * bs
            train_count += bs
            global_step += 1
            if use_tb and (global_step % 50 == 0):
                writer.add_scalar("train/loss_step", loss.item(), global_step)

        train_loss = train_loss_acc / max(1, train_count)
        t1 = time.time()
        print(f"Epoch {epoch} Train MSE: {train_loss:.6f} ({t1-t0:.1f}s)")

        # validation
        model.eval()
        val_acc = 0.0
        val_ct = 0
        with torch.no_grad():
            for batch in val_loader:
                
                preds = model(batch["hist_matrix"].to(args.device),
                              batch["hour_start"].to(args.device),
                              batch["dow"].to(args.device),
                              batch["week_of_year"].to(args.device),
                              batch["lunar_day"].to(args.device),
                              batch["avg_daily_temps_m"].to(args.device),
                              batch["holiday_seq_mn"].to(args.device),
                              batch["itp_id"].to(args.device),
                             # batch["consumer_type_id"].to(args.device),
                             # batch["network_segment_id"].to(args.device)
)
                loss = loss_fn(preds, batch["target"].to(args.device))
                val_acc += loss.item() * preds.size(0)
                val_ct += preds.size(0)
        val_loss = val_acc / max(1, val_ct)
        print(f"Epoch {epoch} Val MSE: {val_loss:.6f}")

        if use_tb:
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("val/epoch_loss", val_loss, epoch)

        scheduler.step(val_loss)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
                    "cfg": cfg.__dict__}, str(last_ckpt))

        stop = early_stopper.step(val_loss, model_state={
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "cfg": cfg.__dict__,
        })
        if val_loss < best_val:
            best_val = val_loss
            print(f"[INFO] New best val {best_val:.6f} at epoch {epoch}. Saved {best_ckpt}")
        if stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

    if use_tb:
        writer.close()
    print("Training finished. Best val:", early_stopper.best)
    print("Best checkpoint:", best_ckpt)
    print("Last checkpoint:", last_ckpt)

# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="./ckpts")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--m", type=int, default=14)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n_itp", type=int, default=10000)
    #p.add_argument("--n_network_segments", type=int, default=50)
    p.add_argument("--n_holiday_ids", type=int, default=6)
    #p.add_argument("--n_consumer_types", type=int, default=5)
    p.add_argument("--early-stop-patience", type=int, default=6)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
