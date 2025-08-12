import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.ticker import FormatStrFormatter

def plot_kaplan_meier(risk, time, indicator, log_dir):
    print("[!] plot kaplan meier curve...")
    indicator = indicator.astype(np.int64)
    # 将数据转换为DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'Survival_Time': time,
        'Event': indicator,
        'Risk_Score': risk
    })

    # 根据风险的中位数将患者分为低风险组和高风险组
    median_risk = np.median(df['Risk_Score'])
    print("median of risks: ", median_risk)
    df['Risk_Group'] = ['Low' if score < median_risk else 'High' for score in df['Risk_Score']]

    # 初始化KaplanMeierFitter
    kmf = KaplanMeierFitter()

        # 设置“毛刺”的样式
    censor_style = {
        'marker': '|',  # 使用竖线表示事件点
        'ms': 6,        # 点的大小
        'mew': 1       # 点的边缘宽度
    }
    # 绘制高风险组的Kaplan-Meier曲线（红色）
    high_risk = df[df['Risk_Group'] == 'High']
    kmf.fit(high_risk['Survival_Time'], event_observed=high_risk['Event'])
    kmf.plot(label='High Risk', ci_show=False, color='#FF001F', show_censors=True,censor_styles=censor_style)

    # 绘制低风险组的Kaplan-Meier曲线（蓝色）
    low_risk = df[df['Risk_Group'] == 'Low']
    try:
        kmf.fit(low_risk['Survival_Time'], event_observed=low_risk['Event'])
    except:
        print(df['Risk_Score'])
        print(low_risk)
        print("error")
        assert False

    kmf.plot(label='Low Risk', ci_show=False, color="#00B7FF", show_censors=True,censor_styles=censor_style)

    # 计算P值
    results = logrank_test(high_risk['Survival_Time'], low_risk['Survival_Time'], event_observed_A=high_risk['Event'], event_observed_B=low_risk['Event'])
    p_value = results.p_value

    # 设置标题和标签
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')

    # 设置y轴刻度只显示一位小数
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # 调整图例的位置
    bottom,top = plt.ylim()
    plt.ylim(bottom-0.065, top)

    plt.legend(loc='upper right', fontsize=10)

    # 将P值文本放在图例下方
    plt.text(x=0.15,y=0.15, s=f'P-value = {p_value:.2e}', transform=plt.gcf().transFigure, fontsize=10)


    # 保存为PNG文件
    plt.savefig(f"{log_dir}/Kaplan-Meier_plot.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # 示例：使用numpy数组输入
    risk = np.array([0.2, 0.8, 0.5, 0.9, 0.3, 0.7, 0.6, 0.4, 0.1, 0.9])
    time = np.array([5, 6, 6, 2, 4, 7, 8, 3, 9, 10])
    indicator = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    # 调用函数，保存Kaplan-Meier曲线
    plot_kaplan_meier(risk, time, indicator, "./")