import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath):
    """加载并预处理火车票数据"""
    try:
        # 加载数据
        logger.info("正在加载数据...")
        data = pd.read_csv(filepath)

        # 数据清洗 - 处理各种表示缺失或无效值的符号
        invalid_values = ['--', '*', '候补', '有', '-----', '--:--', '无', 'null', 'NULL', 'NaN', 'nan']
        data.replace(invalid_values, [np.nan]*len(invalid_values), inplace=True)
        logger.info(f"原始数据形状: {data.shape}")

        # 转换时间特征 - 健壮的处理方式
        def parse_time(time_str):
            try:
                if pd.isna(time_str) or str(time_str).strip() in ['', '-----', '--:--']:
                    return np.nan
                return pd.to_datetime(time_str, format='%H:%M', errors='coerce').time()
            except Exception as e:
                logger.warning(f"无法解析时间 '{time_str}': {str(e)}")
                return np.nan

        data['出发时间'] = data['出发时间'].apply(parse_time)
        data['出发小时'] = data['出发时间'].apply(
            lambda x: x.hour + x.minute/60 if pd.notna(x) else np.nan)

        # 处理历时特征
        def parse_duration(dur):
            if pd.isna(dur):
                return np.nan
            if isinstance(dur, str):
                dur = dur.strip()
                if '次日到达' in dur:
                    try:
                        parts = dur.replace('次日到达', '').split(':')
                        return 24 + float(parts[0]) + float(parts[1])/60
                    except:
                        return np.nan
                try:
                    parts = dur.split(':')
                    return float(parts[0]) + float(parts[1])/60
                except:
                    return np.nan
            return np.nan

        data['历时小时'] = data['历时1'].apply(parse_duration)

        # 处理座位余票 - 转换为数值
        seat_columns = ['商务座特等座', '软卧一等卧', '硬卧二等卧', '硬座']
        for col in seat_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # 用该列的均值填充缺失值
            col_mean = data[col].mean()
            data[col].fillna(col_mean if not np.isnan(col_mean) else 0, inplace=True)
            logger.info(f"列 '{col}' 缺失值填充为: {col_mean if not np.isnan(col_mean) else 0}")

        # 添加星期几特征
        data['查询时间'] = pd.to_datetime(data['查询时间'], errors='coerce')
        data['星期几'] = data['查询时间'].dt.dayofweek  # 周一=0, 周日=6

        # 创建目标变量 - 未来24小时是否有票
        data['未来24小时有票'] = np.where((data[seat_columns] > 0).any(axis=1), 1, 0)

        # 删除仍然含有缺失值的行
        initial_rows = len(data)
        data.dropna(inplace=True)
        logger.info(f"删除包含缺失值的行: {initial_rows - len(data)} 行")

        return data, seat_columns

    except Exception as e:
        logger.error(f"数据加载和预处理失败: {str(e)}")
        raise

def feature_engineering(data, seat_columns):
    """创建用于模型训练的特征"""
    try:
        logger.info("正在进行特征工程...")
        # 基础特征
        features = ['出发小时', '历时小时', '星期几'] + seat_columns

        # 添加交互特征
        data['出发小时_历时'] = data['出发小时'] * data['历时小时']
        data['商务座_软卧'] = data['商务座特等座'] + data['软卧一等卧']
        features.extend(['出发小时_历时', '商务座_软卧'])

        # 添加是否高铁特征
        data['高铁'] = data['车次'].str.startswith('G').astype(int)
        features.append('高铁')

        X = data[features]
        y = data['未来24小时有票']

        return X, y, features

    except Exception as e:
        logger.error(f"特征工程失败: {str(e)}")
        raise

def train_and_evaluate(X, y, features):
    """训练随机森林模型并评估性能"""
    try:
        logger.info("开始模型训练...")
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # 创建预处理和模型的pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1))  # 使用所有CPU核心
        ])

        # 定义超参数网格
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [5, 10, 15, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

        # 网格搜索
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1',
            verbose=1,
            n_jobs=-1)

        logger.info("开始网格搜索...")
        grid_search.fit(X_train, y_train)

        # 最佳模型
        best_model = grid_search.best_estimator_
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

        # 在测试集上评估
        y_pred = best_model.predict(X_test)
        logger.info("\n测试集性能:")
        logger.info(classification_report(y_test, y_pred))
        logger.info(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

        return best_model

    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise

def save_model(model, model_path='models/rf_model.pkl'):
    """保存训练好的模型"""
    try:
        logger.info(f"正在保存模型到 {model_path}...")
        joblib.dump(model, model_path)
        logger.info("模型保存成功")
    except Exception as e:
        logger.error(f"模型保存失败: {str(e)}")
        raise

def main():
    try:
        # 加载和预处理数据
        data, seat_columns = load_and_preprocess_data('data/cunyun.csv')

        # 特征工程
        X, y, features = feature_engineering(data, seat_columns)

        # 训练模型
        model = train_and_evaluate(X, y, features)

        # 保存模型
        save_model(model)

        logger.info("训练流程完成")

    except Exception as e:
        logger.error(f"主流程失败: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())