# data_utils.py
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, cond_data, target_data):
        self.cond_data = cond_data
        self.target_data = target_data

    def __len__(self):
        return len(self.cond_data)

    def __getitem__(self, idx):
        return self.cond_data[idx], self.target_data[idx]
    
def load_and_prepare_data(data, aux_vars, main_vars, 
                          cond_days_main, cond_steps_per_day_main,
                          cond_days_aux, cond_steps_per_day_aux,
                          test_size=0.2):
    cond_main = data["cond"]       # shape: [N, C_main, H, W]
    cond_aux = data["cond_aux"]    # shape: [N, C_aux, H, W]
    target_tensor = data["target"]
    valid_time_array = data["valid_time"]

    # 合併 channel
    cond_tensor = torch.cat([cond_main, cond_aux], dim=1)  # shape: [N, C, H, W]
    num_main_channels = cond_main.shape[1]
    num_aux_channels = cond_aux.shape[1]

    # 過濾條件 (非零比例 > 0.5)
    nonzero_ratio = (target_tensor != 0).float().mean(dim=(1,2,3))
    valid_idx = nonzero_ratio > 0.5
    cond_tensor = cond_tensor[valid_idx]
    target_tensor = target_tensor[valid_idx]
    valid_time_array = valid_time_array[valid_idx.cpu().numpy()]

    # 依時間順序切分
    N = len(cond_tensor)
    split_index = int(N * (1 - test_size))

    cond_train = cond_tensor[:split_index]
    cond_test = cond_tensor[split_index:]
    target_train = target_tensor[:split_index]
    target_test = target_tensor[split_index:]
    time_train = valid_time_array[:split_index]
    time_test = valid_time_array[split_index:]

    # 分開主變數與輔助變數
    cond_main_train = cond_train[:, :num_main_channels, :, :]
    cond_aux_train = cond_train[:, num_main_channels:, :, :]
    cond_main_test = cond_test[:, :num_main_channels, :, :]
    cond_aux_test = cond_test[:, num_main_channels:, :, :]

    # Z-score標準化函數（對多變數 channel）
    def standardize_tensor_by_variable(tensor, var_list, days, steps_per_day):
        C_per_var = days * steps_per_day
        stats = {}
        tensor_std = tensor.clone()
        for i, var in enumerate(var_list):
            start_idx = i * C_per_var
            end_idx = (i + 1) * C_per_var
            sub_tensor = tensor[:, start_idx:end_idx, :, :]
            mean = sub_tensor.mean()
            std = sub_tensor.std()
            tensor_std[:, start_idx:end_idx, :, :] = (sub_tensor - mean) / (std + 1e-8)
            stats[var] = {'mean': mean.item(), 'std': std.item()}
        return tensor_std, stats

    # 主變數標準化（train 統計量）
    cond_main_train_std, main_stats = standardize_tensor_by_variable(
        cond_main_train, main_vars, cond_days_main, cond_steps_per_day_main
    )
    # 用 train 統計量標準化 test
    cond_main_test_std = cond_main_test.clone()
    for i, var in enumerate(main_vars):
        start_idx = i * cond_days_main * cond_steps_per_day_main
        end_idx = (i + 1) * cond_days_main * cond_steps_per_day_main
        mean = main_stats[var]['mean']
        std = main_stats[var]['std']
        cond_main_test_std[:, start_idx:end_idx, :, :] = (cond_main_test[:, start_idx:end_idx, :, :] - mean) / (std + 1e-8)

    # 輔助變數標準化（train 統計量）
    cond_aux_train_std, aux_stats = standardize_tensor_by_variable(
        cond_aux_train, aux_vars, cond_days_aux, cond_steps_per_day_aux
    )
    # 用 train 統計量標準化 test
    cond_aux_test_std = cond_aux_test.clone()
    for i, var in enumerate(aux_vars):
        start_idx = i * cond_days_aux * cond_steps_per_day_aux
        end_idx = (i + 1) * cond_days_aux * cond_steps_per_day_aux
        mean = aux_stats[var]['mean']
        std = aux_stats[var]['std']
        cond_aux_test_std[:, start_idx:end_idx, :, :] = (cond_aux_test[:, start_idx:end_idx, :, :] - mean) / (std + 1e-8)

    # 目標標準化（整體）
    target_mean = target_train.mean()
    target_std = target_train.std()
    target_train_std = (target_train - target_mean) / (target_std + 1e-8)
    target_test_std = (target_test - target_mean) / (target_std + 1e-8)

    # 合併回來
    cond_train_std = torch.cat([cond_main_train_std, cond_aux_train_std], dim=1)
    cond_test_std = torch.cat([cond_main_test_std, cond_aux_test_std], dim=1)

    print("cond_train_main min/max:", cond_main_train_std.min().item(), cond_main_train_std.max().item())
    print("cond_train_aux min/max:", cond_aux_train_std.min().item(), cond_aux_train_std.max().item())
    print("cond_train min/max:", cond_train_std.min().item(), cond_train_std.max().item())
    print("target_train min/max:", target_train_std.min().item(), target_train_std.max().item())

    return (cond_train_std, cond_test_std,
            target_train_std, target_test_std,
            main_stats, aux_stats,
            {"mean": target_mean.item(), "std": target_std.item()},
            time_train, time_test)