import torch
import torch.nn.functional as F
import torch.nn as nn

# --- UNet 定義（與訓練時相同） ---
class UNet(nn.Module):
    def __init__(self, in_channels=24, out_channels=24, cond_channels=72):
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
        d1 = self.down1(torch.cat([x, cond], dim=1))
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        x = d4 + self.res(d4)
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

        if target_shape is not None:
            _, _, H, W = target_shape
            if output.shape[2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        return output

# --- Inference 函數 ---
# 推斷函數 (Algorithm 2, 15 步, 3 幀)
@torch.no_grad()
def ddim_inference(model, cond, beta, device='cuda', eta=0.0):
    """
    DDIM deterministic reverse process (eta=0 means deterministic, eta>0 means stochastic)
    """
    model.eval()
    num_steps = beta.shape[0]
    shape = (1, 24, cond.shape[2], cond.shape[3])
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

    _, cond_test,  _, target_test, cond_min, cond_max, _, _, time_train, time_test = load_and_prepare_data(data)

    import pandas as pd

    # 想找的 target 起始時間（這是要預測的第一個時間點）
    target_date = pd.Timestamp("2015-07-15 00:00")

    # cond 對應的 time_test[i]，加 72 個 3 小時步長，就是 target 的開始時間
    offset = pd.Timedelta(hours=72 * 3)  # 72 個 3 小時 = 9 天

    idx = [i for i, t in enumerate(time_test) if t + offset == target_date]
    if len(idx) == 0:
        raise ValueError("找不到指定的 target 起始時間")
    idx = idx[0]

    # 取得條件輸入資料
    cond_sample = cond_test[idx].unsqueeze(0).to(device)
    
    print(f"對應 cond 時間範圍: {time_test[idx]} ~ {time_test[idx] + pd.Timedelta(hours=3 * 71)}")
    print(f"對應 target 時間範圍: {time_test[idx] + pd.Timedelta(hours=3 * 72)} ~ {time_test[idx] + pd.Timedelta(hours=3 * 95)}")

    # 模型預測
    output = ddim_inference(model, cond_sample, beta, device=device, eta=0.0)

    # Ground Truth
    gt_sample = target_test[idx]  # shape: [24, H, W]

    print("Inference output shape:", output.shape)
    print("gt_sample output shape:", gt_sample.shape)

    # 儲存
    torch.save(output.cpu(), f"./output/generated_output_{idx}.pt")
    torch.save(gt_sample.cpu(), f"./output/ground_truth_{idx}.pt")