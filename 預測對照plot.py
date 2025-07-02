import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

output =  torch.load("./output/generated_output_1756.pt")

# === Ground Truth ===
gt_output = torch.load("./output/ground_truth_1756.pt")

# 將預測拆成三幀（每幀 3 channels）
pred = output[0]  # [9, H, W]

# === 將 24 張圖片按天平均 ===
def average_by_day(tensor_24):  # tensor_24: [24, H, W]
    return torch.stack([
        tensor_24[0:8].mean(dim=0),
        tensor_24[8:16].mean(dim=0),
        tensor_24[16:24].mean(dim=0)
    ])  # return: [3, H, W]

pred_days = average_by_day(pred)
gt_days = average_by_day(gt_output)

# === 畫圖 ===
fig, axs = plt.subplots(2, 3, figsize=(15, 6))  # 2 rows: prediction, ground truth

for i in range(3):
    im = axs[0, i].imshow(pred_days[i].cpu(), cmap='hot')
    axs[0, i].set_title(f"Predicted Day {i+1}")
    axs[0, i].axis('off')

    axs[1, i].imshow(gt_days[i].cpu(), cmap='hot')
    axs[1, i].set_title(f"Ground Truth Day {i+1}")
    axs[1, i].axis('off')

fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
date_str = "2015-07-15_07-18" 
# plt.savefig(f"./output/pred_vs_gt_3day_avg({date_str}).png")

plt.show()