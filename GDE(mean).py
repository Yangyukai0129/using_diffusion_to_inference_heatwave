import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# U-Net 模型 (與圖 6 對應)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, cond_channels=12):
        super(UNet, self).__init__()
        self.down1 = nn.Conv2d(in_channels + cond_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, padding=1)
        self.down4 = nn.Conv2d(256, 384, 3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.res = nn.Sequential(nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 384, 3, padding=1))
        self.up1 = nn.ConvTranspose2d(384, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.up4 = nn.Conv2d(64 + 64, out_channels, 3, padding=1)

    @staticmethod
    def resize_to_match(source_tensor, target_tensor):
        target_h, target_w = target_tensor.shape[2], target_tensor.shape[3]
        return F.interpolate(source_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def forward(self, x, cond, target_shape=None):
        # Down path
        d1 = self.down1(torch.cat([x, cond], dim=1))
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)

        x = d4 + self.res(d4)

        # Up path
        u1 = self.up1(x)
        u1 = UNet.resize_to_match(u1, d3)
        u1 = torch.cat([u1, d3], 1)

        u2 = self.up2(u1)
        u2 = UNet.resize_to_match(u2, d2)
        u2 = torch.cat([u2, d2], 1)

        u3 = self.up3(u2)
        u3 = UNet.resize_to_match(u3, d1)
        u3 = torch.cat([u3, d1], 1)

        output = self.up4(u3)

        # 補齊 or 裁剪，確保與 target 尺寸一致
        if target_shape is not None:
            _, _, H, W = target_shape
            if output.shape[2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        return output

# --- DDIM Inference 函數 ---
@torch.no_grad()
def ddim_inference(model, cond, beta, device='cuda', eta=0.0):
    """
    DDIM deterministic reverse process (eta=0 means deterministic, eta>0 means stochastic)
    """
    model.eval()
    num_steps = beta.shape[0]
    shape = (cond.shape[0], 24, cond.shape[2], cond.shape[3])
    x_t = torch.randn(shape, device=device)  # 初始化噪聲
    cond = cond.to(device)

    # 用訓練時的 β schedule
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    for t in reversed(range(num_steps)):
        # t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        # 預測 noise
        pred_noise = model(x_t, cond)

        # 預測 x0
        sqrt_alpha_t = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

        if t > 0:
            alpha_prev = alpha_cumprod[t - 1].view(-1, 1, 1, 1)
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_cumprod[t]) * (1 - alpha_cumprod[t] / alpha_prev))
            noise = sigma_t * torch.randn_like(x_t) if eta > 0 else 0
            x_t = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev - sigma_t**2) * pred_noise + noise
        else:
            x_t = x0_pred  # 最後一步

    return x_t

# --- MSE 計算 (全天+單天) ---
def compute_daywise_mse(pred, target, device='cuda'):
    pred = pred.to(device)
    target = target.to(device)
    mse_fn = nn.MSELoss(reduction='mean')
    mse_all = mse_fn(pred, target).item()
    mse_day1 = mse_fn(pred[:, 0:8], target[:, 0:8]).item()
    mse_day2 = mse_fn(pred[:, 8:16], target[:, 8:16]).item()
    mse_day3 = mse_fn(pred[:, 16:24], target[:, 16:24]).item()
    return mse_day1, mse_day2, mse_day3, mse_all

# --- 主程式 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    model = UNet(in_channels=24, out_channels=24, cond_channels=72).to(device)
    checkpoint = torch.load("./saved_models/diffusion_model.pth", map_location=device)
    model = UNet(in_channels=24, out_channels=24, cond_channels=72).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])  # 載入模型參數
    beta = checkpoint["beta"].to(device)                  # 載入訓練時的 beta
    model.eval()

    # 載入資料
    from data_utils import load_and_prepare_data
    import torch

    data = torch.load("data/cond_target_time_day.pt", weights_only=False)
    _, cond_test, _, target_test, _, _, _, _, _, time_test = load_and_prepare_data(data)

    cond_test = cond_test.to(device)
    target_test = target_test.to(device)

    import pandas as pd

    # === 從 CSV 讀取 valid_time 列表 ===
    csv_path = "./data/valid_time.csv"  # 你的 CSV 路徑
    df = pd.read_csv(csv_path)
    time_strs = df['valid_time'].tolist()  # 讀取 valid_time 欄位

        # === 找出對應 index ===
    import numpy as np

    offset = np.timedelta64(72 * 3, 'h')  # 9 天（72 個 3 小時）

    # 把 cond_time + 9 天 統一轉成日期（精度到天）
    cond_dates = (time_test + offset).astype('datetime64[s]')

    # 找到時間對應的 index
    indices = []
    for t in time_strs:
        target_date = np.datetime64(t).astype('datetime64[s]')  # 只取日期部分
        matches = np.where(cond_dates == target_date)[0]
        if len(matches) > 0:
            indices.extend(matches)  # 收集所有符合的 index
            print(f"✅ 找到 {len(matches)} 個符合 {t} 的 index")
        else:
            print(f"⚠️ Warning: {t} 不在 target 起始時間中，已跳過")

    # === 子集資料 ===
    cond_subset = cond_test[indices]
    target_subset = target_test[indices]

    # === 多次 sampling ===
    N = 15  # Sampling 次數
    all_samples = []
    print(f"開始生成，每筆資料生成 {N} 次...")
    for i in tqdm(range(N), desc="Sampling", unit="sample"):
        generated_i = ddim_inference(model, cond_subset, beta, device=device, eta=0.0)
        all_samples.append(generated_i.detach())

        # 單次 MSE
        d1, d2, d3, all_mse = compute_daywise_mse(generated_i, target_subset)
        tqdm.write(f"Sampling {i+1} MSE ➝ Day1: {d1:.6f}, Day2: {d2:.6f}, Day3: {d3:.6f}, All: {all_mse:.6f}")

    all_samples = torch.stack(all_samples)  # shape: [N, batch, 24, H, W]
    gde_mean = all_samples.mean(dim=0)

    # GDE-mean MSE
    # 如果是生成全部測試集把target_subset改成target_test
    d1, d2, d3, all_mse = compute_daywise_mse(gde_mean, target_subset, device=device)
    print(f"\nGDE-mean MSE ➝ Day1: {d1:.6f}, Day2: {d2:.6f}, Day3: {d3:.6f}, All: {all_mse:.6f}")

    # 儲存結果
    # torch.save(all_samples, "./output/generated_all_samples.pt")
    # torch.save(gde_mean, "./output/generated_gde_mean.pt")
    # torch.save(target_test.cpu(), "./output/ground_truth_all.pt")
    # print("已儲存所有生成結果和 GDE-mean")

