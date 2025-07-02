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
def inference(model, cond, num_steps=15, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1,  24,  cond.shape[2], cond.shape[3]).to(device)
        cond = cond.to(device)
        print("x shape:", x.shape)
        print("cond shape:", cond.shape)
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t / num_steps], device=device).float()
            alpha = 1 - t_tensor
            z = torch.randn_like(x) if t > 0 else 0
            pred_noise = model(x, cond)
            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / ((1 - alpha).sqrt()) * pred_noise) + (1 - alpha).sqrt() * z
        return x

# --- 主程式 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    model = UNet(in_channels=24, out_channels=24, cond_channels=72).to(device)
    model.load_state_dict(torch.load("./saved_models/diffusion_model.pth", map_location=device))
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
    output = inference(model, cond_sample)

    # Ground Truth
    gt_sample = target_test[idx]  # shape: [24, H, W]

    print("Inference output shape:", output.shape)
    print("gt_sample output shape:", gt_sample.shape)

    # 儲存
    torch.save(output.cpu(), f"./output/generated_output_{idx}.pt")
    torch.save(gt_sample.cpu(), f"./output/ground_truth_{idx}.pt")