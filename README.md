# using_diffusion_to_predict_climate
1. 資料轉換.py：將資料轉換成tensor，並且condition input為前9天(8時間段x9)、target為後3天(8時間段x3)
2. train_diffusion_model.py：模型訓練
3. inference_only.py：進行生成
4. 預測對照plot.py：將實際溫度和生成溫度進行圖示化

data_util：資料拆分# using_diffusion_to_inference_heatwave
