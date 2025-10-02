import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train_transformer_holiday import SyntheticDataset, CollateWithNormalization, Config, FullForecastModel#TransformerHoliday



# -------------------------------
# Настройки демо
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path("./ckpts")
checkpoint = out_dir / "best_model.pt"

m, n = 14, 3  # прошлые дни и дни прогноза
batch_size = 4  # сколько ИТП визуализировать одновременно

# -------------------------------
# Загружаем синтетические данные для демо
# -------------------------------
cfg = Config()
cfg.m = m
cfg.n = n

dataset = SyntheticDataset(cfg, N=20)  # возьмем 20 ИТП для демонстрации

# Загружаем нормализаторы
normalizers_path = out_dir / "normalizers.json"
collate_fn = CollateWithNormalization(normalizers_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# -------------------------------
# Загружаем модель
# ----------------------------------

model = FullForecastModel(cfg).to(device)
ckpt = torch.load(checkpoint, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# -------------------------------
# Предсказания
# -------------------------------
with torch.no_grad():
    for batch in loader:
        hist = batch["hist_matrix"].to(device)
        hour_start = batch["hour_start"].to(device)
        dow = batch["dow"].to(device)
        week = batch["week_of_year"].to(device)
        lunar = batch["lunar_day"].to(device)
        temps = batch["avg_daily_temps_m"].to(device)
        holiday_seq = batch["holiday_seq_mn"].to(device)
        itp_id = batch["itp_id"].to(device)
        #consumer_id = batch["consumer_type_id"].to(device)
        #seg_id = batch["network_segment_id"].to(device)
        target = batch["target"].cpu().numpy()

        preds = model(hist, hour_start, dow, week, lunar, temps, holiday_seq, itp_id) #, consumer_id, seg_id
        preds = preds.cpu().numpy()  # [B, n*24]

        # -------------------------------
        # Визуализация по каждому ИТП в батче
        # -------------------------------
        for i in range(2): #preds.shape[0]
            plt.figure(figsize=(12,4))
            hours = np.arange(n*24)
            plt.plot(hours, target[i], label="Target (True)", color="blue")
            plt.plot(hours, preds[i], label="Prediction", color="red", linestyle="--")
            plt.title(f"ITP Demo {i} - Predicted vs True water consumption")
            plt.xlabel("Hour (next n*24)")
            plt.ylabel("Water consumption")
            plt.legend()
            plt.grid(True)
            plt.show()
