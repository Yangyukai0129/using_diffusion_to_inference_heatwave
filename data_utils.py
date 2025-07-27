# data_utils.py
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, cond_data, target_data, cond_min=None, cond_max=None, target_min=None, target_max=None):
        self.cond_data = cond_data
        self.target_data = target_data

        # Normalization stats
        self.cond_min = cond_min
        self.cond_max = cond_max
        self.target_min = target_min
        self.target_max = target_max
        self.eps = 1e-6

    def __len__(self):
        return len(self.cond_data)

    def __getitem__(self, idx):
        cond = self.cond_data[idx]
        target = self.target_data[idx]

        if self.cond_min is not None:
            cond = (cond - self.cond_min) / (self.cond_max - self.cond_min + self.eps)
        if self.target_min is not None:
            target = (target - self.target_min) / (self.target_max - self.target_min)

        return cond, target

def load_and_prepare_data(data, test_size=0.2):
    cond_tensor = data["cond"]
    target_tensor = data["target"]
    valid_time_array = data["valid_time"]

    # 過濾
    nonzero_ratio = (target_tensor != 0).float().mean(dim=(1, 2, 3))
    valid_idx = nonzero_ratio > 0.5
    cond_tensor = cond_tensor[valid_idx]
    target_tensor = target_tensor[valid_idx]
    valid_time_array = valid_time_array[valid_idx.numpy()]

    # 切分
    N = len(cond_tensor)
    split_index = int(N * (1 - test_size))

    cond_train = cond_tensor[:split_index]
    cond_test = cond_tensor[split_index:]
    target_train = target_tensor[:split_index]
    target_test = target_tensor[split_index:]
    time_train = valid_time_array[:split_index]
    time_test = valid_time_array[split_index:]

    # 正規化參數
    cond_min = cond_train.min()
    cond_max = cond_train.max()
    target_min = target_train.min()
    target_max = target_train.max()

    return cond_train, cond_test, target_train, target_test, cond_min, cond_max, target_min, target_max, time_train, time_test

def denormalize(data, data_min, data_max, eps=1e-6):
    """
    將 min-max normalized 資料還原回原始範圍
    data: Tensor，已標準化 (0~1)
    data_min, data_max: Tensor 或 float，原始資料的最小/最大值
    """
    return data * (data_max - data_min + eps) + data_min