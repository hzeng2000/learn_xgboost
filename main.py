import pandas as pd
from utils import read_data, feature_engineering, train_model, predict_future
import joblib

def main():
    # 1. 读取数据
    file_path = "/home/hzeng/prj/learn_xgboost/tourist_data.xlsx"  # 替换为实际路径
    data = read_data(file_path, 600)

    # 2. 特征工程
    data = feature_engineering(data)

    # 3. 定义特征和目标变量
    features = [
        'month',              # 月份
        'weekday',            # 工作日（0-6）
        'is_weekend',         # 是否为周末
        '节假日',             # 是否为节假日
        '周末',               # 是否为周末
        '连续节假日天数',      # 连续节假日的天数
        'lag_1',              # 前一天流量
        'rolling_mean_7'      # 过去7天的平均流量
    ]
    target = '原始数据'

    # 4. 训练模型
    model = train_model(data, features, target, tuning=True)
    joblib.dump(model, "trained_model.pkl")

    # 5. 准备未来预测数据
    future_data = pd.DataFrame({
        'month': [1],  # 假设为2025年1月
        'weekday': [3],  # 假设为星期三
        'is_weekend': [0],
        '节假日': [0],
        '周末': [0],
        '连续节假日天数': [0],
        'lag_1': [1545],  # 假设上一天的流量为1545
        'rolling_mean_7': [2000],  # 假设过去7天的平均流量为2000
    })

    # 6. 预测未来数据
    future_prediction = predict_future(model, future_data)
    print(f"2025年1月的预测流量: {future_prediction[0]}")

if __name__ == "__main__":
    main()