import pandas as pd
import joblib  # 用于加载保存的模型
from utils import feature_engineering, predict_future

def predict_from_excel(model_path, file_path, start_row, end_row):
    """
    从 Excel 文件中读取指定行，生成特征并进行预测。
    :param model_path: str, 保存的模型文件路径
    :param file_path: str, Excel 文件路径
    :param start_row: int, 起始行（从 0 开始计数）
    :param end_row: int, 结束行（不包括该行）
    :return: None
    """
    # 1. 读取指定行的数据
    data = pd.read_excel(file_path, skiprows=range(1, start_row), nrows=end_row - start_row)
    data['日期'] = pd.to_datetime(data['日期'])

    # 检查读取的数据是否为空
    if data.empty:
        print("指定的行范围没有数据，请检查 start_row 和 end_row 参数！")
        return

    # 2. 特征工程
    data = feature_engineering(data)

    # 3. 加载训练好的模型
    model = joblib.load(model_path)

    # 4. 定义特征列表（与训练时一致）
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

    # 5. 检查特征是否存在
    if not all(f in data.columns for f in features):
        raise ValueError(f"数据缺少必要的特征，请检查输入文件。需要的特征包括: {features}")

    # 6. 进行预测
    predictions = model.predict(data[features])  # 使用模型直接预测

    # 7. 输出结果
    for i, pred in enumerate(predictions):
        date = data.iloc[i]['日期']  # 获取当前行的日期
        print(f"预测结果 - 日期 {date.date()} 流量: {pred:.2f}")


if __name__ == "__main__":
    # 配置路径和参数
    model_path = "trained_model.pkl"  # 替换为你的模型文件路径
    file_path = "/home/hzeng/prj/learn_xgboost/tourist_data.xlsx"  # 替换为你的 Excel 文件路径
    start_row = 633  # 起始行（从第 638 行开始）
    end_row = 645    # 结束行（到第 645 行，不包括第 645 行）

    # 调用预测函数
    predict_from_excel(model_path, file_path, start_row, end_row)