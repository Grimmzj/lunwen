import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

# 读取CSV数据
df = pd.read_csv('data/cunyun.csv')

# 数据预处理函数 - 路线分析
def preprocess_route_data(df):
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

# 数据预处理函数 - 剩余票数分析
def preprocess_ticket_data(df):
    # 只分析可以预订的车次
    df = df[df['可以预定'] == '预订'].copy()

    # 将票数信息转换为数值
    seat_types = ['商务座特等座', '软卧一等卧', '硬卧二等卧', '硬座']

    for seat in seat_types:
        # 将"候补"、"有"等转换为数值
        df[seat] = df[seat].replace({
            '候补': 0,
            '有': 1,
            '*': 0,
            '--': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '16': 16,
            '18': 18
        })
        # 将字符串数字转换为数值
        df[seat] = pd.to_numeric(df[seat], errors='coerce').fillna(0)

    # 计算总剩余票数
    df['总剩余票数'] = df[seat_types].sum(axis=1)

    # 添加路线信息
    df['路线'] = df['出发站'] + '-' + df['到达站']

    # 按剩余票数排序
    return df.sort_values('总剩余票数', ascending=False)

@app.route('/')
def index():
    # 路线分析数据
    route_stats = preprocess_route_data(df)
    top_routes = route_stats.head(10)

    # 剩余票数分析数据
    ticket_stats = preprocess_ticket_data(df)
    top_tickets = ticket_stats.head(10)

    # 准备路线分析图表数据
    route_names = top_routes['路线'].tolist()
    bookable_counts = top_routes['可预订车次数量'].tolist()
    total_counts = top_routes['总车次数量'].tolist()
    bookable_ratio = top_routes['可预订比例'].tolist()

    # 准备剩余票数图表数据
    train_numbers = top_tickets['车次'].tolist()
    ticket_counts = top_tickets['总剩余票数'].tolist()
    routes_for_tickets = top_tickets['路线'].tolist()
    seat_details = []

    for _, row in top_tickets.iterrows():
        seat_details.append({
            '商务座特等座': int(row['商务座特等座']),
            '软卧一等卧': int(row['软卧一等卧']),
            '硬卧二等卧': int(row['硬卧二等卧']),
            '硬座': int(row['硬座'])
        })

    return render_template('index.html',
                           routes_data=top_routes.to_dict('records'),
                           route_names=route_names,
                           bookable_counts=bookable_counts,
                           total_counts=total_counts,
                           bookable_ratio=bookable_ratio,
                           tickets_data=top_tickets[['车次', '路线', '总剩余票数']].to_dict('records'),
                           train_numbers=train_numbers,
                           ticket_counts=ticket_counts,
                           routes_for_tickets=routes_for_tickets,
                           seat_details=seat_details)

@app.route('/analysis1')
def analysis1():
    # 获取可预订车次数据
    ticket_stats = preprocess_ticket_data(df)
    # 获取更多数据用于分析（前50条）
    top_tickets = ticket_stats.head(50)

    # 准备数据
    tickets_data = top_tickets[['车次', '路线', '历时1', '出发时间']].to_dict('records')
    seat_details = []

    for _, row in top_tickets.iterrows():
        seat_details.append({
            '商务座特等座': int(row['商务座特等座']),
            '软卧一等卧': int(row['软卧一等卧']),
            '硬卧二等卧': int(row['硬卧二等卧']),
            '硬座': int(row['硬座'])
        })

    return render_template('index1.html',
                           tickets_data=tickets_data,
                           seat_details=seat_details)

@app.route('/analysis2')
def analysis2():
    # 获取可预订车次数据
    ticket_stats = preprocess_ticket_data(df)
    # 获取更多数据用于分析（前50条）
    top_tickets = ticket_stats.head(50)

    # 准备数据
    tickets_data = top_tickets[['车次', '路线', '出发时间']].to_dict('records')
    seat_details = []

    for _, row in top_tickets.iterrows():
        seat_details.append({
            '商务座特等座': int(row['商务座特等座']),
            '软卧一等卧': int(row['软卧一等卧']),
            '硬卧二等卧': int(row['硬卧二等卧']),
            '硬座': int(row['硬座'])
        })

    return render_template('index2.html',
                           tickets_data=tickets_data,
                           seat_details=seat_details)

if __name__ == '__main__':
    app.run(debug=True)