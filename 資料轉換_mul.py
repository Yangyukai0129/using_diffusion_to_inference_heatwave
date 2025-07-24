import numpy as np
import torch
import xarray as xr

def create_cond_target_pairs_multiscale(ds_main, ds_aux_list, main_vars, aux_vars, 
                                        cond_days_main=9, cond_steps_per_day_main=8,
                                        cond_days_aux=3, cond_steps_per_day_aux=8,
                                        target_days=3, target_steps_per_day=8):
    """
    建立主要變數和多個輔助變數的條件張量，以及目標變數張量，並對齊時間。

    輔助變數時間軸假設與主變數相同，但抓取的時間長度不同。
    """

    # === 基本參數 ===
    time_dim = 'valid_time'
    T_main = ds_main.sizes[time_dim]
    H, W = ds_main.sizes['latitude'], ds_main.sizes['longitude']

    cond_len_main = cond_days_main * cond_steps_per_day_main
    cond_len_aux = cond_days_aux * cond_steps_per_day_aux
    target_len = target_days * target_steps_per_day

    # 样本數
    N = T_main - cond_len_main - target_len + 1
    if N <= 0:
        raise ValueError("❌ 資料長度不足以建立樣本，請檢查天數與時間步數設定")

    # 直接切出 valid_times：每筆樣本目標時間的起始時間點
    valid_times = ds_main[time_dim].values[cond_len_main : cond_len_main + N]

    # === 結果列表 ===
    cond_main_list, cond_aux_list, target_list = [], [], []

    for i in range(N):
        # === 主變數 ===
        cond_main_vars_list = []
        for var in main_vars:
            arr = ds_main[var].values  # [T, H, W]
            cond_main_slice = arr[i : i + cond_len_main]
            cond_main_vars_list.append(cond_main_slice)
        cond_main_arr = np.concatenate(cond_main_vars_list, axis=0)  # [C_main, H, W]

        # === 輔助變數（用主變數 idx 切片） ===
        cond_aux_vars_list = []
        start_idx_aux = i + cond_len_main - cond_len_aux
        end_idx_aux = i + cond_len_main
        # for j, var in enumerate(aux_vars):
        #     arr = ds_aux_list[j][var].values  # [T, H, W]
        #     cond_aux_slice = arr[start_idx_aux:end_idx_aux]
        #     cond_aux_vars_list.append(cond_aux_slice)

        for var in aux_vars:
            for ds in ds_aux_list:
                if var in ds:
                    arr = ds[var].values  # [T, H, W]
                    break
            else:
                raise KeyError(f"❌ {var} not found in any ds_aux_list Dataset")

            cond_aux_slice = arr[start_idx_aux:end_idx_aux]
            cond_aux_vars_list.append(cond_aux_slice)

        cond_aux_arr = np.concatenate(cond_aux_vars_list, axis=0)  # [C_aux, H, W]

        # === 目標變數 ===
        target_vars_list = []
        for var in main_vars:
            arr = ds_main[var].values
            target_slice = arr[i + cond_len_main : i + cond_len_main + target_len]
            target_vars_list.append(target_slice)
        target_arr = np.concatenate(target_vars_list, axis=0)  # [C_target, H, W]

        # === 加入結果 ===
        cond_main_list.append(cond_main_arr)
        cond_aux_list.append(cond_aux_arr)
        target_list.append(target_arr)

    # === 組成張量 ===
    cond_main_tensor = torch.from_numpy(np.stack(cond_main_list)).float()
    cond_aux_tensor = torch.from_numpy(np.stack(cond_aux_list)).float()
    target_tensor = torch.from_numpy(np.stack(target_list)).float()

    print(f"✅ 最終樣本數: {len(valid_times)}")
    print(f"cond_main shape: {cond_main_tensor.shape}")
    print(f"cond_aux shape:  {cond_aux_tensor.shape}")
    print(f"target shape:    {target_tensor.shape}")

    return cond_main_tensor, cond_aux_tensor, target_tensor, valid_times

# 範例使用：
if __name__ == "__main__":
    # 讀取不同nc檔
    ds_main = xr.open_dataset("./data/merged_1965_2024.nc")  # 例如溫度
    ds_aux1 = xr.open_dataset("./download_nc/merged_5deg_wind(1965-2024).nc")  # 例如濕度
    # ds_aux2 = xr.open_dataset("./download_nc/merged_5deg_v(1965-2024).nc")  # 例如風速

    main_vars = ["t"]             # 主變數名
    aux_vars = ["wind_speed","wind_direction"] # 輔助變數名
    ds_aux_list = [ds_aux1]

    cond_main, cond_aux, target, valid_times = create_cond_target_pairs_multiscale(
        ds_main, ds_aux_list, main_vars, aux_vars,
        cond_days_main=9, cond_steps_per_day_main=8,
        cond_days_aux=3, cond_steps_per_day_aux=8,
        target_days=3, target_steps_per_day=8
    )

    # 7. 儲存
    torch.save({
        "cond": cond_main,
        "cond_aux": cond_aux,
        "target": target,
        "valid_time": valid_times
    }, "./data/cond_aux(wind)_target_time_day.pt")

    print("儲存完成：cond_aux(wind)_target_time.pt")
    print("cond_main shape:", cond_main.shape)
    print("cond_aux shape:", cond_aux.shape)
    print("target shape:", target.shape)
    print("valid_times shape:", valid_times.shape)