import torch
import torch.nn as nn
from unet import UNet, train
import torch.optim as optim
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 建立模型
    model = UNet(in_channels=24, out_channels=24, cond_channels=72, time_dim=32).to(device)

    # 1. 取得檔案清單
    train_files, test_files = prepare_file_list("data/npy_files_2")

    # 2. 計算 normalization 統計數值
    cond_mean, cond_std, target_mean, target_std = compute_mean_std(train_files)

    # 3. 建立 Dataset
    train_dataset = LazyWeatherDataset(train_files, cond_mean, cond_std, target_mean, target_std)
    # test_dataset = LazyWeatherDataset(test_files, cond_mean, cond_std, target_mean, target_std)

    # 4. DataLoader（開多 worker）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.L1Loss()
    train_loss_history = []

    # 第一階段訓練
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_loss_history, beta, alpha, alpha_cumprod = train(
        model, train_loader, 40, device,
        optimizer, criterion,
        train_loss_history,
        use_checkpoint=True  # 開啟梯度檢查點
    )

    # 釋放資源
    torch.save({
        "model_state_dict": model.state_dict(),
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod
    }, "./saved_models/temp_stage1.pth")
    torch.cuda.empty_cache()

    # 重新建立模型並載入權重
    ckpt_stage1 = torch.load("./saved_models/temp_stage1.pth", map_location=device)
    beta = ckpt_stage1["beta"].to(device)
    alpha = ckpt_stage1["alpha"].to(device)
    alpha_cumprod = ckpt_stage1["alpha_cumprod"].to(device)

    model.load_state_dict(ckpt_stage1["model_state_dict"])

    # 第二階段 fine-tune
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    train_loss_history, beta, alpha, alpha_cumprod = train(
        model, train_loader, 10, device,
        optimizer, criterion,
        train_loss_history,
        use_checkpoint=True,
        beta=beta,  # ✅ 直接沿用
        alpha=alpha,
        alpha_cumprod=alpha_cumprod
    )

    # 最終儲存
    torch.save({
        "model_state_dict": model.state_dict(),
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "optimizer_state_dict": optimizer.state_dict(),
    }, "./saved_models/diffusion_model_region.pth")
    print("模型已儲存完成")

if __name__ == "__main__":
    main()


    # import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # import matplotlib.pyplot as plt

    # # === 繪圖 ===
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_loss_history, label='Train Loss')
    # plt.plot(val_loss_history, label='Val Loss')
    # plt.axvline(40, color='gray', linestyle='--', label='Fine-tune start')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Train vs Val Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("loss_curve_step15.png")
    # plt.show()