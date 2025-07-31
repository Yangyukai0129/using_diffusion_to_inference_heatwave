import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# 用於提供模型一種「知道時間在哪裡」的方式
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),  # 或 nn.SiLU()
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class DownBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        skip1 = self.res1(x)    # 第一層 skip output
        skip2 = self.res2(skip1)  # 第二層 skip output
        out = skip2
        return out, skip1, skip2  # 回傳所有你需要的 skip 給對應的 UpBlock
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        skip1 = self.res1(x)    # 第一層 skip output
        skip2 = self.res2(skip1)  # 第二層 skip output
        out = self.pool(skip2)  # pool 後的輸出
        return out, skip1, skip2  # 回傳所有你需要的 skip 給對應的 UpBlock
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip1_channels, skip2_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.res1 = ResidualBlock(in_channels + skip1_channels, out_channels)
        self.res2 = ResidualBlock(out_channels + skip2_channels, out_channels)

    def forward(self, x, skip1, skip2):
        x = self.upsample(x)
        x = torch.cat([skip1, x], dim=1)
        x = self.res1(x)
        x = torch.cat([skip2, x], dim=1)
        x = self.res2(x)
        return x
    
class SkipConnection(nn.Module):
    def __init__(self, mode="concat"):
        super(SkipConnection, self).__init__()
        self.mode = mode

    def forward(self, encoder_feat, decoder_feat):
        if self.mode == "concat":
            return torch.cat([decoder_feat, encoder_feat], dim=1)  # channel 方向 concat
        elif self.mode == "add":
            return decoder_feat + encoder_feat
        else:
            raise ValueError("Unsupported skip connection mode")

# U-Net 模型 (與圖 6 對應)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, cond_channels=12, time_dim=32):
        super(UNet, self).__init__()
        self.out_channels = out_channels
        self.time_dim = time_dim
        # 時間嵌入層
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, 32),  # 從 time_dim 映射到 32 維
            nn.SiLU(),               # 激活函數
            nn.Linear(32, 32)        # 再次映射
        )

        self.target_shape = None
        self.skip = SkipConnection("concat")
        self.pool = nn.AvgPool2d(2, 2)

        # 下採樣層
        # 用 ResidualBlock 替代原本的單層 Conv2d
        self.down = nn.Conv2d(in_channels + cond_channels, 64, kernel_size=3, padding=1)
        self.time = ResidualBlock(32, 32)
        self.down1 = DownBlock1(96, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 384)
        self.res = ResidualBlock(384, 384)
        # 上採樣層
        self.up1 = UpBlock(384, 384, 384, 256)  # 對應 skip7, skip8
        self.up2 = UpBlock(256, 256, 256, 128)  # 對應 skip5, skip6
        self.up3 = UpBlock(128, 128, 128, 64)     # 對應 skip3, skip4

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def resize_to_match(source_tensor, target_tensor):
        target_h, target_w = target_tensor.shape[2], target_tensor.shape[3]
        return F.interpolate(source_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def forward(self, x, cond, t, beta):
        device = x.device
        B, _, H, W = x.shape
        embedding_dim = self.time_dim  # 這邊對應你初始化UNet時傳入的time_dim參數
        beta_t = beta[t].to(device)
        noise_variance = beta_t  # [batch_size]

        # 1. 產生 sinusoidal embedding 向量
        t_emb = sinusoidal_embedding(noise_variance, embedding_dim, device)  # [B, 32]
        # print(t_emb.shape)
        t_emb = self.time_embed(t_emb)  # [batch_size, 32]
        # print(t_emb.shape)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 32, 1, 1]
        # print(t_emb.shape)
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))  # [batch_size, 32, H, W]
        # print(t_emb.shape)

        # 2. concat cond + noisy x → 做卷積
        x = torch.cat([x, cond], dim=1)  # [B, in + cond, H, W]
        x = self.down(x)                # [B, 64, H, W]
        # print("x",x.shape)

        # residual時間embedding
        t_emb = self.time(t_emb)

        # 3. concat 時間嵌入
        x = torch.cat([x, t_emb], dim=1)     # [B, 96, H, W]
        # print("x",x.shape)
       
        # Down path (多層 DownBlock 回傳 skip1, skip2)
        d1, skip1, skip2 = self.down1(x)        # [B, 64, H, W]    # (H, W)      → skip1, skip2
        # print("d1",d1.shape)
        d2, skip3, skip4 = self.down2(d1)  # channels=128   # (H/2, W/2)  → skip3, skip4
        # print("d2",d2.shape)
        d3, skip5, skip6 = self.down3(d2)  # channels=256   # (H/4, W/4)  → skip5, skip6
        # print("d3",d3.shape)
        d4, skip7, skip8 = self.down4(d3)   # (H/8, W/8)  → skip7, skip8
        # print("d4",d4.shape)
        # down4 是 ResidualBlock，沒有skip，若想有skip，可以改寫或用兩層ResidualBlock
        bottleneck = self.res(d4)           # (H/8, W/8)                 # channels=384

        # Up path：
        u1 = self.up1(bottleneck, skip7, skip8)  # (H/4, W/4)
        # print("u1",u1.shape)
        # u1 = self.skip(u1, d3)
        # print("u1",u1.shape)
        u2 = self.up2(u1, skip5, skip6)          # (H/2, W/2)
        # print("u2",u2.shape)
        # u2 = self.skip(u2, d2)
        # print("u2",u2.shape)
        u3 = self.up3(u2, skip3, skip4)          # (H, W)
        # print("u3",u3.shape)
        # u3 = self.skip(u3, d1)
        # print("u3",u3.shape)

        output = self.final_conv(u3)
        # print("output",output.shape)

        # 補齊 or 裁剪，確保與 target 尺寸一致
        if hasattr(self, 'target_shape') and self.target_shape is not None: 
            _, _, H, W = self.target_shape
            if output.shape[2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        return output

# 訓練函數 (Algorithm 1)
from torch.utils.checkpoint import checkpoint

def train(model, train_loader, num_epochs, device,
          optimizer, criterion, train_loss_history,
          use_checkpoint=False,
          beta=None, alpha=None, alpha_cumprod=None):

    # 如果沒有外部傳入，才自己生成
    if beta is None or alpha is None or alpha_cumprod is None:
        beta = torch.linspace(1e-4, 0.02, steps=1000)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

    beta = beta.to(device)
    alpha = alpha.to(device)
    alpha_cumprod = alpha_cumprod.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for cond, target, time in train_loader:
            cond = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            batch_size = cond.shape[0]
            t = torch.randint(0, 1000, (batch_size,), device=device)

            noise = torch.randn_like(target)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod[t])[:, None, None, None]
            print(f"target shape: {target.shape}")
            print(f"noise shape: {noise.shape}")
            print(f"sqrt_alpha_cumprod_t shape: {sqrt_alpha_cumprod_t.shape}")
            print(f"sqrt_one_minus_alpha_cumprod_t shape: {sqrt_one_minus_alpha_cumprod_t.shape}")
            x_t = sqrt_alpha_cumprod_t * target + sqrt_one_minus_alpha_cumprod_t * noise

            optimizer.zero_grad(set_to_none=True)
            if use_checkpoint:
                output = checkpoint(model, x_t, cond, t, beta, use_reentrant=False)
            else:
                output = model(x_t, cond, t, beta)

            loss = criterion(output, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}")

    return train_loss_history, beta.cpu(), alpha.cpu(), alpha_cumprod.cpu()

@torch.no_grad()
def ddim_inference(model, cond, beta, device, eta=0.0, num_steps=15):
    """
    DDIM 推理流程
    model: UNet 模型
    cond: 條件輸入 (B, cond_channels, H, W)
    beta: 訓練時的 beta (Tensor)
    eta: 控制隨機性 (0 = deterministic)
    num_steps: 推理步數
    """
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

    # 初始化噪聲
    shape = (cond.shape[0], model.out_channels, cond.shape[2], cond.shape[3])
    x_t = torch.randn(shape, device=device)

    # 推理步長
    step_size = 1000 // num_steps
    timesteps = list(range(0, 1000, step_size))[::-1]

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((cond.shape[0],), t, device=device, dtype=torch.long)

        # 模型預測噪聲
        pred_noise = model(x_t, cond, t_tensor, beta)

        # DDIM 公式
        alpha_t = alpha_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        x0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_next = alpha_cumprod[t_next]
            sigma_t = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            noise = sigma_t * torch.randn_like(x_t)
            x_t = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next - sigma_t**2) * pred_noise + noise
        else:
            x_t = x0_pred  # 最後一步

    return x_t

