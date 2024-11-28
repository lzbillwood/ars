import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

# 设置 matplotlib 字体以支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_data(uploaded_file):
    """加载CSV文件并处理数据"""
    try:
        df = pd.read_csv(uploaded_file, encoding='gbk')
        df = df.fillna(0)  # 将缺失值填充为0
        df = df.astype(bool)  # 确保数据框中的所有值都是0或1，并转换为布尔类型
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None

def generate_rules(df, min_support=0.1, min_confidence=0.7):
    """生成频繁项集和关联规则"""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by='lift', ascending=False)
    return rules

def filter_rules(rules, min_antecedents, min_lift):
    """根据前项数和提升度筛选规则"""
    return rules[
        (rules['antecedents'].apply(lambda x: len(x) >= min_antecedents)) &
        (rules['lift'] >= min_lift)
    ]

def create_network_graph(rules_filtered):
    """创建网络图并保存为HTML文件"""
    G = nx.DiGraph()
    for i, row in rules_filtered.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        G.add_edge(antecedents, consequents, weight=row['lift'])

    net = Network(notebook=True, height="750px", width="100%")
    net.from_nx(G)
    net.save_graph("arn3.html")
    return "arn3.html"

def main():
    st.title("关联规则挖掘与可视化")

    # 上传CSV文件
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            rules = generate_rules(df)

            # 显示关联规则
            st.write("关联规则:")
            st.write(rules)

            # 用户选择前项数和提升度
            min_antecedents = st.slider("选择前项数的最小值", min_value=1, max_value=5, value=1)
            min_lift = st.slider("选择提升度的最小值", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

            # 筛选前项数和提升度
            rules_filtered = filter_rules(rules, min_antecedents, min_lift)

            # 显示筛选后的关联规则
            st.write("筛选后的关联规则:")
            st.write(rules_filtered)

            # 创建网络图
            html_file = create_network_graph(rules_filtered)
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=750)

if __name__ == "__main__":
    main()