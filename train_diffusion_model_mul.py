import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# 訓練函數 (Algorithm 1)
def train(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, train_loss_history=None, val_loss_history=None):
    if train_loss_history is None: train_loss_history = []
    if val_loss_history is None: val_loss_history = []

    # === 定義 beta schedule ===
    T_steps = 1000  # diffusion steps
    beta = torch.linspace(1e-4, 0.02, T_steps).to(device)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)  # shape: [T_steps]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            # === 取離散時間步 t ===
            t = torch.randint(0, T_steps, (target.size(0),), device=device).long()
            alpha_t = alpha_cumprod[t].view(-1, 1, 1, 1)  # broadcast to target shape

            # print("target shape:", target.shape)            
            # print("cond shape:", cond.shape)

            # alpha = alpha.view(target.size(0), 1, 1, 1)  # 把 alpha 從 [batch_size] 變成 [batch_size, 1, 1, 1]
            # === 加噪 ===
            noise = torch.randn_like(target)
            # print("noise shape:", noise.shape)
            noisy_target = torch.sqrt(alpha_t) * target + torch.sqrt(1 - alpha_t) * noise

            pred_noise = model(noisy_target, cond, target_shape=target.shape)
            loss = criterion(pred_noise, noise)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * cond.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cond, target in val_loader:
                cond, target = cond.to(device), target.to(device)
                t = torch.randint(0, T_steps, (target.size(0),), device=device).long()
                alpha_t = alpha_cumprod[t].view(-1, 1, 1, 1)

                noise = torch.randn_like(target)
                noisy_target = torch.sqrt(alpha_t) * target + torch.sqrt(1 - alpha_t) * noise

                pred_noise = model(noisy_target, cond, target_shape=target.shape)
                loss = criterion(pred_noise, noise)
                val_loss += loss.item() * cond.size(0)

        val_loss /= len(val_loader.dataset)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_loss_history, val_loss_history, beta, alpha, alpha_cumprod

# 主程序
if __name__ == "__main__":

    # 模型與裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=24, out_channels=24, cond_channels=120).to(device)

    from data_utils_mul import WeatherDataset, load_and_prepare_data
    from torch.utils.data import DataLoader

    data = torch.load("data/cond_aux(wind)_target_time_day.pt", weights_only=False)

    cond_train, cond_test, target_train, target_test, _, _, _, time_train, time_test = load_and_prepare_data(data,aux_vars=['wind_speed','wind_direction'],main_vars=['t'],cond_days_main=9,cond_steps_per_day_main=8,cond_days_aux=3,cond_steps_per_day_aux=8)

    train_dataset = WeatherDataset(cond_train, target_train)
    test_dataset = WeatherDataset(cond_test, target_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 儲存歷史紀錄
    train_loss_history = []
    val_loss_history = []

    criterion = nn.MSELoss() # mse
   
    # === 初始訓練階段 ===
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_loss_history, val_loss_history, beta, alpha, alpha_cumprod = train(
        model, train_loader, test_loader, num_epochs=40, device=device,
        optimizer=optimizer, criterion=criterion,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history
    )

    # === fine-tuning 階段（繼續接上去） ===
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    train_loss_history, val_loss_history, beta, alpha, alpha_cumprod = train(
        model, train_loader, test_loader, num_epochs=10, device=device,
        optimizer=optimizer, criterion=criterion,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history
    ) 
    
    # 儲存模型
    torch.save({
    "model_state_dict": model.state_dict(),
    "beta": beta.cpu(),                   # 儲存 beta
    "alpha": alpha.cpu(),                 # 儲存 alpha
    "alpha_cumprod": alpha_cumprod.cpu(), # 儲存 alpha_cumprod
    "optimizer_state_dict": optimizer.state_dict(), # 可選：儲存 optimizer
    }, "./saved_models/diffusion_model_mul(wind).pth")
    print("模型已儲存完成")

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import matplotlib.pyplot as plt

    # === 繪圖 ===
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.axvline(40, color='gray', linestyle='--', label='Fine-tune start')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("loss_curve_mul.png")
    plt.show()