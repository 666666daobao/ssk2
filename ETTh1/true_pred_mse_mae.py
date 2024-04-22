import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

ETTh1_root_path = r'F:\Transformer\LTSF_Linear_main\LTSF_Linear_main\dataset\ett'
ETTh1_path = r'ETTh1_scaler.csv'
ETTh1_source_path = os.path.join(ETTh1_root_path, ETTh1_path)
# 读取 ettmm1 数据
# ettmm1_data = pd.read_csv(ETTm1_source_path)
etth1_data = pd.read_csv(ETTh1_path)

pred_root_path = r'F:\Transformer\LTSF_Linear_main\LTSF_Linear_main'
pred_path = r'new_predictions_etth1.csv'
pred_source_path = os.path.join(pred_root_path, pred_path)

# 读取新的预测结果数据
new_predictions = pd.read_csv(pred_path)

# 选择 ettmm1 数据的倒数 66610 行到最后一行
# etth1_tail = etth1_data.iloc[66608:]
ettmm1_tail = ettmm1_data.iloc[14348:]

# 计算 MSE 和 MAE
mse = mean_squared_error(etth1_tail['OT'], new_predictions['OT'])
mae = mean_absolute_error(etth1_tail['OT'], new_predictions['OT'])

print("MSE:", mse)
print("MAE:", mae)

mse_total = 0
mae_total = 0
num_columns = len(ettmm1_tail.columns)

for column in ettmm1_tail.columns[1:]:
    mse_column = mean_squared_error(ettmm1_tail[column], new_predictions[column])
    mae_column = mean_absolute_error(ettmm1_tail[column], new_predictions[column])
    mse_total += mse_column
    mae_total += mae_column

mse_avg = mse_total / num_columns
mae_avg = mae_total / num_columns

print("Average MSE:", mse_avg)
print("Average MAE:", mae_avg)