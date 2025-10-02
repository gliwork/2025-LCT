"""
forecast_with_holiday_transformer.py

Variant of the forecasting model where the holiday sequence (length m+n)
is processed by a Transformer encoder instead of flattened embeddings.

Compatibility: inputs and dataset format are same as before.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    m = 14
    n = 3
    emb_k = 128
    emb_p = 64
    decoder_hidden = 256
    dropout = 0.2

    # categorical sizes (fill before training)
    n_itp = 10000
    #n_consumer_types = 5      #число разных типов потребителей
    n_holiday_ids = 8         # number of distinct holiday categories
    #n_network_segments = 50

    # embedding sizes
    emb_itp = 32
    #emb_consumer = 8
    #emb_segment = 8
    emb_holiday = 16          # must be divisible by n_heads for Transformer
    holiday_nheads = 4
    holiday_layers = 2

    batch_size = 128
    lr = 1e-3
    epochs = 30
    seed = 42

# ---------------------------
# Holiday Transformer encoder
# ---------------------------
class PositionalEncodingLearned(nn.Module):
    """Learned positional embeddings for sequence length up to max_len."""
    def __init__(self, emb_dim: int, max_len: int = 128):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.max_len = max_len
        self.emb_dim = emb_dim

    def forward(self, seq_len: int, device=None):
        """
        returns positional embeddings of shape (seq_len, emb_dim)
        """
        if seq_len > self.max_len:
            # extend embedding table if necessary (rare)
            raise ValueError(f"seq_len {seq_len} > max_len {self.max_len}")
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return self.pos_emb(positions)  # (seq_len, emb_dim)

class HolidayTransformerEncoder(nn.Module):
    """
    Embeds holiday token ids with embedding -> add learned position embeddings ->
    TransformerEncoder layers -> returns pooled vector (mean over seq) length emb_holiday.
    """
    def __init__(self, n_holiday_ids: int, emb_holiday: int, num_layers: int = 2, n_heads: int = 4, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        assert emb_holiday % n_heads == 0, "emb_holiday must be divisible by n_heads"
        self.emb = nn.Embedding(n_holiday_ids, emb_holiday)
        self.pos = PositionalEncodingLearned(emb_dim=emb_holiday, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_holiday, nhead=n_heads, dropout=dropout, dim_feedforward=emb_holiday*4, activation='relu', batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, holiday_seq_ids: torch.Tensor):
        """
        holiday_seq_ids: (B, seq_len) LongTensor
        returns pooled vector: (B, emb_holiday)
        """
        B, seq_len = holiday_seq_ids.shape
        device = holiday_seq_ids.device
        # token embeddings (B, seq_len, emb)
        x = self.emb(holiday_seq_ids)  # (B, seq_len, emb)
        # add pos: make pos shape (1, seq_len, emb) then broadcast
        pos = self.pos(seq_len, device=device).unsqueeze(0)  # (1, seq_len, emb)
        x = x + pos  # (B, seq_len, emb)
        x = self.dropout(x)
        # Transformer expects (seq_len, B, emb) OR if using batch_first=False (default) we must permute
        x = x.permute(1, 0, 2)  # (seq_len, B, emb)
        # no src_key_padding_mask used (assumes full length)
        out = self.transformer(x)  # (seq_len, B, emb)
        out = out.permute(1, 0, 2)  # (B, seq_len, emb)
        # pooling: mean over seq dimension
        pooled = out.mean(dim=1)  # (B, emb)
        return pooled


# ---------------------------
# Updated TabularEncoder using HolidayTransformerEncoder
# ---------------------------
class TabularEncoderTransformer(nn.Module):
    def __init__(self, m: int, n: int, emb_p: int, cfg: Config):
        super().__init__()
        self.m = m
        self.n = n
        self.cfg = cfg

        # categorical embeddings
        self.emb_itp = nn.Embedding(cfg.n_itp, cfg.emb_itp)
        #self.emb_consumer = nn.Embedding(cfg.n_consumer_types, cfg.emb_consumer)
        #self.emb_segment = nn.Embedding(cfg.n_network_segments, cfg.emb_segment)

        # holiday transformer
        self.holiday_transformer = HolidayTransformerEncoder(n_holiday_ids=cfg.n_holiday_ids,
                                                             emb_holiday=cfg.emb_holiday,
                                                             num_layers=cfg.holiday_layers,
                                                             n_heads=cfg.holiday_nheads,
                                                             dropout=cfg.dropout,
                                                             max_len=m + n + 8)

        # numeric scalars: hour_start, dow, week_of_year, lunar_day  => 4
        num_scalars = 4
        temps_dim = m  # avg_daily_temps_m

        # after holiday transformer pooling we get emb_holiday vector
        holiday_pooled_dim = cfg.emb_holiday

        input_dim = num_scalars + temps_dim + cfg.emb_itp +  holiday_pooled_dim #cfg.emb_consumer + cfg.emb_segment +

        hidden = max(128, emb_p * 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, emb_p),
            nn.ReLU(),
            nn.LayerNorm(emb_p),
        )

    def forward(self, hour_start, dow, week_of_year, lunar_day,
                avg_daily_temps_m, holiday_seq_mn, itp_id):#, consumer_type_id, network_segment_id
        """
        Inputs:
         - hour_start, dow, week_of_year, lunar_day: (B,1) floats
         - avg_daily_temps_m: (B,m) floats
         - holiday_seq_mn: (B, m+n) long (ids)
         - itp_id, consumer_type_id, network_segment_id: (B,) long
        """
        B = hour_start.shape[0]
        emb_itp = self.emb_itp(itp_id)                       # (B, emb_itp)
       # emb_cons = self.emb_consumer(consumer_type_id)       # (B, emb_consumer)
       # emb_seg = self.emb_segment(network_segment_id)       # (B, emb_segment)

        # holiday transformer pooled vector
        holiday_pooled = self.holiday_transformer(holiday_seq_mn)  # (B, emb_holiday)

        scalars = torch.cat([hour_start, dow, week_of_year, lunar_day], dim=1)  # (B,4)
        x = torch.cat([scalars, avg_daily_temps_m, emb_itp, holiday_pooled], dim=1)#, emb_cons, emb_seg
        out = self.mlp(x)
        return out


# ---------------------------
# Conv encoder and decoder (unchanged, copied minimal versions)
# ---------------------------
class ConvHistoryEncoder(nn.Module):
    def __init__(self, m: int, emb_k: int, dropout: float = 0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, min(4, max(1, m // 2)))),
        )
        dummy_out_h, dummy_out_w = 4, min(4, max(1, m // 2))
        flattened = 96 * dummy_out_h * dummy_out_w
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, emb_k),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_k),
        )

    def forward(self, x):
        z = self.conv_block(x)
        out = self.fc(z)
        return out

class ForecastDecoder(nn.Module):
    def __init__(self, emb_k, emb_p, n, dropout=0.2, hidden=256):
        super().__init__()
        out_dim = n * 24
        self.net = nn.Sequential(
            nn.Linear(emb_k + emb_p, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, out_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, out_dim), out_dim)
        )

    def forward(self, emb_conv, emb_tab):
        x = torch.cat([emb_conv, emb_tab], dim=1)
        out = self.net(x)
        return out

class FullForecastModelTransformerHoliday(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.conv_enc = ConvHistoryEncoder(m=cfg.m, emb_k=cfg.emb_k, dropout=cfg.dropout)
        self.tab_enc = TabularEncoderTransformer(m=cfg.m, n=cfg.n, emb_p=cfg.emb_p, cfg=cfg)
        self.decoder = ForecastDecoder(emb_k=cfg.emb_k, emb_p=cfg.emb_p, n=cfg.n, dropout=cfg.dropout, hidden=cfg.decoder_hidden)

    def forward(self, hist_matrix,
                hour_start, dow, week_of_year, lunar_day,
                avg_daily_temps_m, holiday_seq_mn,
                itp_id):#, consumer_type_id, network_segment_id
        emb_c = self.conv_enc(hist_matrix)
        emb_t = self.tab_enc(hour_start, dow, week_of_year, lunar_day,
                             avg_daily_temps_m, holiday_seq_mn,
                             itp_id)#, consumer_type_id, network_segment_id
        out = self.decoder(emb_c, emb_t)
        return out

# ---------------------------
#Инициализация модели: instantiate model
# ---------------------------
if __name__ == "__main__":
    cfg = Config()
    # set categorical sizes to realistic values
    cfg.n_itp = 10
    #cfg.n_network_segments = 50
    cfg.n_holiday_ids = 6
    #cfg.n_consumer_types = 5
    # ensure emb_holiday divisible by heads
    cfg.emb_holiday = 16
    cfg.holiday_nheads = 4
    model = FullForecastModelTransformerHoliday(cfg).to(DEVICE)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model params:", total)
    # quick forward with random tensors
    B = 2
    hist = torch.randn(B, 1, 24, cfg.m, device=DEVICE)
    hour = torch.rand(B,1, device=DEVICE)
    dow = torch.rand(B,1, device=DEVICE)
    week = torch.rand(B,1, device=DEVICE)
    lunar = torch.rand(B,1, device=DEVICE)
    temps = torch.randn(B, cfg.m, device=DEVICE)
    holidays = torch.randint(0, cfg.n_holiday_ids, (B, cfg.m + cfg.n), device=DEVICE)
    itp_ids = torch.randint(0, cfg.n_itp, (B,), device=DEVICE)
    #cons_ids = torch.randint(0, cfg.n_consumer_types, (B,), device=DEVICE)
    #seg_ids = torch.randint(0, cfg.n_network_segments, (B,), device=DEVICE)
    out = model(hist, hour, dow, week, lunar, temps, holidays, itp_ids) #, cons_ids, seg_ids
    print("out shape:", out.shape)  # (B, n*24)
