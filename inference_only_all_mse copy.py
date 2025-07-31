import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm

def sinusoidal_embedding(t, dim, device):
    """
    生成正弦嵌入表示
    t: [batch_size]，時間步長
    dim: 嵌入維度（例如 32）
    返回: [batch_size, dim] 的嵌入張量
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]  # [batch_size, half_dim]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [batch_size, dim]
    return emb

# U-Net 模型 (與圖 6 對應)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, cond_channels=12, time_dim=32):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        # 時間嵌入層
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, 64),  # 從 time_dim 映射到 64 維
            nn.SiLU(),               # 激活函數
            nn.Linear(64, 64)        # 再次映射
        )
        # 下採樣層
        self.down1 = nn.Conv2d(in_channels + cond_channels + 64, 64, 3, padding=1)  # 增加時間嵌入通道
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, padding=1)
        self.down4 = nn.Conv2d(256, 384, 3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.res = nn.Sequential(nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 384, 3, padding=1))
        # 上採樣層
        self.up1 = nn.ConvTranspose2d(384, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.up4 = nn.Conv2d(64 + 64, out_channels, 3, padding=1)

    @staticmethod
    def resize_to_match(source_tensor, target_tensor):
        target_h, target_w = target_tensor.shape[2], target_tensor.shape[3]
        return F.interpolate(source_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def forward(self, x, cond, t):
        # 生成時間嵌入
        t_emb = sinusoidal_embedding(t, self.time_dim, x.device)  # [batch_size, time_dim]
        t_emb = self.time_embed(t_emb)  # [batch_size, 64]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 64, 1, 1]
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))  # [batch_size, 64, H, W]

        # 與輸入和條件拼接
        x = torch.cat([x, cond, t_emb], dim=1)  # [batch_size, in_channels + cond_channels + 64, H, W]

        # Down path
        d1 = self.down1(x)
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
        if hasattr(self, 'target_shape') and self.target_shape is not None:
            _, _, H, W = self.target_shape
            if output.shape[2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        return output

@torch.no_grad()
def ddim_inference(model, cond, beta, num_steps=None, device='cuda', eta=0.0):
    """
    DDIM deterministic reverse process (eta=0 means deterministic, eta>0 means stochastic)
    Args:
        model: UNet 模型
        cond: 條件輸入 [batch_size, cond_channels, H, W]
        beta: 訓練時的 beta 調度 [T_steps]
        num_steps: DDIM 步數（若為 None，則使用 beta 的長度）
        device: 設備
        eta: 噪聲權重 (0 for deterministic)
    Returns:
        generated: 生成的圖像 [batch_size, out_channels, H, W]
    """
    model.eval()
    batch_size = cond.shape[0]
    if num_steps is None:
        num_steps = beta.shape[0]  # 使用訓練時的步數
    shape = (batch_size, 3, cond.shape[2], cond.shape[3])  # 根據 out_channels=24 設置
    x_t = torch.randn(shape, device=device)  # 初始化噪聲

    cond = cond.to(device)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    # DDIM 時間步索引（非均勻間隔）
    t_steps = torch.linspace(0, num_steps - 1, num_steps, device=device, dtype=torch.long)
    for t in tqdm(reversed(range(num_steps)), desc="DDIM Inference"):
        t_tensor = t_steps[t].repeat(batch_size)  # [batch_size]

        # 預測 noise，使用時間嵌入
        pred_noise = model(x_t, cond, t_tensor)

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

# --- 主程式 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(1)
    # 載入模型
    checkpoint = torch.load("./saved_models/diffusion_model_region_step1000(1D).pth", map_location=device)
    model = UNet(in_channels=3, out_channels=3, cond_channels=3, time_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])  # 載入模型參數
    beta = checkpoint["beta"].to(device)                  # 載入訓練時的 beta
    model.eval()
    print(2)
    # 載入資料
    from data_utils import load_and_prepare_data, denormalize
    import torch

    data = torch.load("data/subset_36N-60N_-12E-12E(1D).pt", weights_only=False)
    print(2.1)
    cond_train, cond_test, target_train, target_test, cond_mean, cond_std, target_mean, target_std, time_train, time_test = load_and_prepare_data(data)
    print(3)
    cond_test = cond_test.to(device)
    target_test = target_test.to(device)

    import pandas as pd

    # === 從 CSV 讀取 valid_time 列表 ===
    csv_path = "./data/valid_time(20).csv"  # 你的 CSV 路徑
    df = pd.read_csv(csv_path)
    time_strs = df['valid_time'].tolist()  # 讀取 valid_time 欄位
    print(4)
    # === 找出對應 index ===
    import numpy as np

    # 把 cond_time + 9 天 統一轉成日期（精度到天）
    cond_dates = (time_test).astype('datetime64[s]')
    print(5)
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

    # === 全部測試集生成 ===
    print("開始生成全部測試集...")
    # 如果是生成全部測試集把cond_subset改成cond_test
    # 加上進度條
    generated_list = []
    for i in tqdm(range(cond_subset.shape[0]), desc="Generating Samples"):
        gen = ddim_inference(model, cond_subset[i:i+1], beta, device=device, eta=0.0)
        generated_list.append(gen)
    generated = torch.cat(generated_list, dim=0)

    print("生成完成，開始計算 MSE...")
    # === 三天拆分 ===
    def split_days(tensor):
        return tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
    
    # 反標準化生成結果與真實目標
    # 如果是生成全部測試集把target_subset改成target_test
    generated_denorm = denormalize(generated, target_mean, target_std)
    target_denorm = denormalize(target_subset, target_mean, target_std)

    # 如果是生成全部測試集把target_subset改成target_test
    gen_day1, gen_day2, gen_day3 = split_days(generated_denorm)
    gt_day1, gt_day2, gt_day3 = split_days(target_denorm)


    
    # === 計算 MSE ===
    # 如果是生成全部測試集把target_subset改成target_test
    mse_fn = nn.MSELoss(reduction='mean')
    mse_day1 = mse_fn(gen_day1, gt_day1).item()
    mse_day2 = mse_fn(gen_day2, gt_day2).item()
    mse_day3 = mse_fn(gen_day3, gt_day3).item()
    mse_all = mse_fn(generated_denorm, target_denorm).item()

    print(f"MSE (Day 1): {mse_day1:.6f}")
    print(f"MSE (Day 2): {mse_day2:.6f}")
    print(f"MSE (Day 3): {mse_day3:.6f}")
    print(f"MSE (All 3 Days): {mse_all:.6f}")

    # 儲存結果
    # torch.save(generated.cpu(), "./output/generated_all.pt")
    # torch.save(target_test.cpu(), "./output/ground_truth_all.pt")
    # print("已儲存所有生成結果和 Ground Truth")
