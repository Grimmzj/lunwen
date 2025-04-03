from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import random
import logging
from threading import Lock

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
model_lock = Lock()

# 加载模型
def load_model(model_path='models/rf_model.pkl'):
    try:
        logger.info(f"正在加载模型从 {model_path}...")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

# 初始化模型
try:
    model = load_model()
except:
    model = None

# 模拟车次数据
def generate_train_data():
    trains = []
    for i in range(10):
        hour = random.randint(0, 23)
        minute = random.choice(['00', '15', '30', '45'])
        trains.append({
            'id': f'G{random.randint(1000, 9999)}',
            'from': random.choice(['北京', '上海', '广州', '深圳', '成都']),
            'to': random.choice(['长沙', '武汉', '杭州', '南京', '重庆']),
            'departure': f"{hour}:{minute}",
            'duration': f"{random.randint(1, 8)}小时{random.randint(0, 59)}分",
            'business_seats': random.randint(0, 20),
            'first_seats': random.randint(0, 50),
            'second_seats': random.randint(0, 100)
        })
    return sorted(trains, key=lambda x: x['departure'])

# 生成24小时预测数据
def generate_prediction_data(train_info):
    try:
        with model_lock:
            if model is None:
                raise ValueError("模型未加载")

            # 准备特征数据
            features = model.named_steps['rf'].feature_names_in_
            input_data = {
                '出发小时': float(train_info['departure'].split(':')[0]),
                '历时小时': float(train_info['duration'].split('小时')[0]),
                '星期几': datetime.now().weekday(),
                '商务座特等座': train_info['business_seats'],
                '软卧一等卧': train_info['first_seats'],
                '硬卧二等卧': train_info['second_seats'],
                '出发小时_历时': float(train_info['departure'].split(':')[0]) * float(train_info['duration'].split('小时')[0]),
                '商务座_软卧': train_info['business_seats'] + train_info['first_seats'],
                '高铁': 1 if train_info['id'].startswith('G') else 0
            }

            # 确保特征顺序正确
            X_pred = pd.DataFrame([input_data])[features]

            # 标准化特征
            X_pred_scaled = model.named_steps['scaler'].transform(X_pred)

            # 预测
            pred = model.named_steps['rf'].predict(X_pred_scaled)
            proba = model.named_steps['rf'].predict_proba(X_pred_scaled)

            # 生成24小时预测数据
            hours = [f"{i:02d}:00" for i in range(24)]
            current_hour = datetime.now().hour

            # 模拟预测结果 - 实际应用中应使用真实预测逻辑
            business = [max(0, train_info['business_seats'] - i) for i in range(24)]
            first = [max(0, train_info['first_seats'] - i*2) for i in range(24)]
            second = [max(0, train_info['second_seats'] - i*5) for i in range(24)]

            return {
                'hours': hours,
                'data': {
                    'business': business,
                    'first': first,
                    'second': second
                },
                'prediction': int(pred[0]),
                'probability': float(proba[0][1])
            }

    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        # 返回模拟数据作为后备
        hours = [f"{i:02d}:00" for i in range(24)]
        return {
            'hours': hours,
            'data': {
                'business': [random.randint(0, 20) for _ in range(24)],
                'first': [random.randint(0, 50) for _ in range(24)],
                'second': [random.randint(0, 100) for _ in range(24)]
            },
            'prediction': random.randint(0, 1),
            'probability': random.random()
        }

@app.route('/')
def index():
    try:
        trains = generate_train_data()
        return render_template('index3.html', trains=trains)
    except Exception as e:
        logger.error(f"首页渲染失败: {str(e)}")
        return render_template('error.html', message="系统初始化失败"), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        train_info = {
            'id': request.form.get('train_id'),
            'from': request.form.get('from'),
            'to': request.form.get('to'),
            'departure': request.form.get('departure'),
            'duration': request.form.get('duration'),
            'business_seats': int(request.form.get('business_seats', 0)),
            'first_seats': int(request.form.get('first_seats', 0)),
            'second_seats': int(request.form.get('second_seats', 0))
        }

        result = generate_prediction_data(train_info)
        return jsonify(result)

    except Exception as e:
        logger.error(f"预测请求处理失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)