import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import xgboost as xgb

def read_data(file_path, n_rows=None):
    """
    读取数据并进行初步处理，支持读取表的前 n 行。
    :param file_path: str, Excel 文件路径
    :param n_rows: int, 读取的行数。如果为 None，则读取整个表。
    :return: DataFrame
    """
    data = pd.read_excel(file_path, nrows=n_rows)
    data['日期'] = pd.to_datetime(data['日期'])
    return data

def feature_engineering(data):
    """
    对数据进行特征工程，包括时间特征提取、历史特征生成和滚动特征计算。
    :param data: DataFrame, 原始数据
    :return: DataFrame, 添加了特征的数据
    """
    # 时间特征
    data['month'] = data['日期'].dt.month  # 月份
    data['weekday'] = data['日期'].dt.weekday  # 一周中的第几天（0=周一，6=周日）
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)  # 是否为周末

    # 滞后特征
    data['lag_1'] = data['原始数据'].shift(1)  # 前一天的流量

    # 滚动特征
    data['rolling_mean_7'] = data['原始数据'].rolling(window=7).mean()  # 过去7天的平均流量

    # 删除无法计算特征的行（如前7天的滚动平均值为 NaN）
    data = data.dropna()

    return data



def train_model(data, features, target, tuning=False):
    """
    训练 XGBoost 模型，并支持超参数调优。
    :param data: DataFrame, 包含特征和目标变量的数据
    :param features: list, 特征列名
    :param target: str, 目标变量列名
    :param tuning: bool, 是否启用超参数调优
    :return: 已训练的 XGBoost 模型
    """
    X = data[features]
    y = data[target]

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    if tuning:
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],         # 树的数量
            'learning_rate': [0.01, 0.05, 0.1],     # 学习率
            'max_depth': [3, 5, 7],                 # 树的深度
            'subsample': [0.6, 0.8, 1.0],           # 数据采样比例
            'colsample_bytree': [0.6, 0.8, 1.0],    # 特征采样比例
            'reg_alpha': [0, 0.1, 1],               # L1 正则化
            'reg_lambda': [1, 10, 100]              # L2 正则化
        }

        # 初始化模型
        model = xgb.XGBRegressor(random_state=42)

        # 使用网格搜索调优（可以改成 RandomizedSearchCV 提高效率）
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # 使用负均方误差作为评估标准
            cv=3,                             # 3折交叉验证
            verbose=1,
            n_jobs=-1                         # 使用所有可用内核
        )

        # 执行网格搜索
        grid_search.fit(X_train, y_train)

        # 最佳参数和模型
        best_params = grid_search.best_params_
        print(f"最佳参数: {best_params}")

        model = grid_search.best_estimator_

    else:
        # 如果不进行调优，使用默认参数进行训练
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"模型评估 - RMSE: {rmse}")

    return model

def predict_future(model, future_data):
    """
    使用训练好的模型预测未来数据。
    :param model: 已训练的 XGBoost 模型
    :param future_data: DataFrame, 包含未来预测所需特征的数据
    :return: 预测值
    """
    return model.predict(future_data)