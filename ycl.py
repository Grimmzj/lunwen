import pandas as pd
from flask import Flask, render_template

# 读取CSV数据
df = pd.read_csv('data/cunyun.csv')

# 数据预处理
def preprocess_data(df):
    # 筛选可预订的车次
    df_bookable = df[df['可以预定'] == '预订'].copy()

    # 创建"路线"列，格式为"出发站-到达站"
    df_bookable['路线'] = df_bookable['出发站'] + '-' + df_bookable['到达站']

    # 统计各路线可预订车次数量
    route_counts = df_bookable['路线'].value_counts().reset_index()
    route_counts.columns = ['路线', '可预订车次数量']

    # 统计各路线总车次数量
    df['路线'] = df['出发站'] + '-' + df['到达站']
    total_counts = df['路线'].value_counts().reset_index()
    total_counts.columns = ['路线', '总车次数量']

    # 合并两个统计结果
    merged = pd.merge(route_counts, total_counts, on='路线', how='outer').fillna(0)
    merged['可预订比例'] = (merged['可预订车次数量'] / merged['总车次数量'] * 100).round(1)

    return merged.sort_values('可预订车次数量', ascending=False)

# 预处理数据
route_stats = preprocess_data(df)
top_routes = route_stats.head(10)
app = Flask(__name__)

@app.route('/')
def index():
    # 准备ECharts需要的数据
    route_names = top_routes['路线'].tolist()
    bookable_counts = top_routes['可预订车次数量'].tolist()
    total_counts = top_routes['总车次数量'].tolist()

    # 准备可预订比例数据
    bookable_ratio = top_routes['可预订比例'].tolist()

    return render_template('index.html',
                           route_names=route_names,
                           bookable_counts=bookable_counts,
                           total_counts=total_counts,
                           bookable_ratio=bookable_ratio)

if __name__ == '__main__':
    app.run(debug=True)