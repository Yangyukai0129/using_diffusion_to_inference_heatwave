import numpy as np
import torch
import xarray as xr

# Step A：切片函式
def create_cond_target_pairs(data, cond_days=9, target_days=3):
    T, H, W = data.shape
    N = T - cond_days - target_days + 1
    cond_list, target_list = [], []
    for i in range(N):
        cond = data[i : i + cond_days]
        target = data[i + cond_days : i + cond_days + target_days]
        cond_list.append(cond)
        target_list.append(target)
    cond_array = np.stack(cond_list)     # [N, 9, H, W]
    target_array = np.stack(target_list) # [N, 3, H, W]
    return cond_array, target_array

# Step B：多變數堆疊
def stack_multivariable_time_windows(ds, var_list, cond_days=9, target_days=3):
    cond_tensors, target_tensors = [], []
    for var in var_list:
        data = ds[var].values  # shape: (T, H, W)
        cond, target = create_cond_target_pairs(data, cond_days, target_days)
        cond_tensors.append(cond)
        target_tensors.append(target)
    cond_all = np.concatenate(cond_tensors, axis=1)     # [N, C_cond, H, W]
    target_all = np.concatenate(target_tensors, axis=1) # [N, C_target, H, W]
    return torch.from_numpy(cond_all).float(), torch.from_numpy(target_all).float()

# === 主程式 ===
# 1. 讀取原始資料
ds = xr.open_dataset("./data/merged_1965_2024.nc")
temp = ds['t']
cond_steps = 72   # 9 天 × 8 次/天
target_steps = 24 # 3 天 × 8 次/天

# 2. 計算每日最大氣溫（resample）
temp_daily = temp.resample(valid_time='1D').max()
temp_daily = temp_daily.chunk({'valid_time': -1})  # 讓計算在後續不受 chunk 限制

# 3. 更新 Dataset
ds_daily = xr.Dataset({'t': temp})  # 這裡可加入其他變數如你之後有用到 ["u", "v", "z", ...]

# 4. 擷取 daily valid_time
valid_times = temp['valid_time'].values  # shape: [14720]（假設從 1965/7 到 2024/8）

# 5. 製作 cond/target tensor
cond_vars = ["t"]
cond_tensor, target_tensor = stack_multivariable_time_windows(ds_daily, cond_vars, cond_days=cond_steps, target_days=target_steps)

# 確保 target_times 和 cond_tensor 數量一致
N = cond_tensor.shape[0]  # ← 保證一致，不要重新計算
target_times = valid_times[cond_steps : cond_steps + N]

# === 篩選 6–8 月的資料 ===
target_months = target_times.astype('datetime64[M]').astype(int) % 12 + 1
summer_mask = np.isin(target_months, [6, 7, 8])

cond_tensor = cond_tensor[summer_mask]
target_tensor = target_tensor[summer_mask]
target_times = target_times[summer_mask]

# 7. 儲存
torch.save({
    "cond": cond_tensor,
    "target": target_tensor,
    "valid_time": target_times
}, "./data/cond_target_time_day.pt")

print("儲存完成：cond_target_time.pt")
print("cond_tensor:", cond_tensor.shape)
print("target_tensor:", target_tensor.shape)
print("valid_time:", target_times.shape)