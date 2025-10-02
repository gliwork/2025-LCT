import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

class WaterConsumptionDataset(Dataset):
    def __init__(self, dataset_path, format="parquet", normalizers_path=None):
        """
        Args:
            dataset_path: path to Parquet file or pickled dataset list
            format: "parquet" or "pkl"
            normalizers_path: optional path to JSON with mean/std for features
        """
        self.format = format
        self.normalizers = None
        if normalizers_path is not None:
            import json
            with open(normalizers_path, "r") as f:
                self.normalizers = json.load(f)

        if format == "parquet":
            self.df = pd.read_parquet(dataset_path)
            # If hist_matrix / target are stored as lists, convert to np.array
            self.df['hist_matrix'] = self.df['hist_matrix'].apply(lambda x: np.array(x, dtype=np.float32))
            self.df['avg_daily_temps_m'] = self.df['avg_daily_temps_m'].apply(lambda x: np.array(x, dtype=np.float32))
            self.df['holiday_seq_mn'] = self.df['holiday_seq_mn'].apply(lambda x: np.array(x, dtype=np.int64))
            self.df['target'] = self.df['target'].apply(lambda x: np.array(x, dtype=np.float32))
        elif format == "pkl":
            with open(dataset_path, "rb") as f:
                self.dataset_list = pickle.load(f)
        else:
            raise ValueError("format must be 'parquet' or 'pkl'")

    def __len__(self):
        if self.format == "parquet":
            return len(self.df)
        else:
            return len(self.dataset_list)

    def __getitem__(self, idx):
        if self.format == "parquet":
            row = self.df.iloc[idx]
            sample = {
                "itp_id": np.array([row["itp_id"]], dtype=np.int64),
                "hist_matrix": row["hist_matrix"],  # [24, m]
                "avg_daily_temps_m": row["avg_daily_temps_m"],  # [m]
                "holiday_seq_mn": row["holiday_seq_mn"],  # [m+n]
                "hour_start": np.array([row["hour_start"]], dtype=np.float32),
                "dow": np.array([row["dow"]], dtype=np.float32),
                "week_of_year": np.array([row["week_of_year"]], dtype=np.float32),
                "lunar_day": np.array([row["lunar_day"]], dtype=np.float32),
                "consumer_type_id": np.array([row.get("consumer_type_id", 0)], dtype=np.int64),
                "network_segment_id": np.array([row.get("network_segment_id", 0)], dtype=np.int64),
                "target": row["target"]
            }
        else:
            sample = self.dataset_list[idx]

        # Apply normalization if normalizers provided
        if self.normalizers:
            hist_mean = np.array(self.normalizers["hist_matrix"]["mean"], dtype=np.float32)
            hist_std = np.array(self.normalizers["hist_matrix"]["std"], dtype=np.float32)
            sample["hist_matrix"] = (sample["hist_matrix"] - hist_mean) / (hist_std + 1e-8)

            temps_mean = np.array(self.normalizers["avg_daily_temps_m"]["mean"], dtype=np.float32)
            temps_std = np.array(self.normalizers["avg_daily_temps_m"]["std"], dtype=np.float32)
            sample["avg_daily_temps_m"] = (sample["avg_daily_temps_m"] - temps_mean) / (temps_std + 1e-8)

        return sample


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # For Parquet dataset
    dataset = WaterConsumptionDataset("dataset.parquet", format="parquet", normalizers_path="normalizers.json")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in loader:
        print(batch["hist_matrix"].shape, batch["target"].shape)
        break
