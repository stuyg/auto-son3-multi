import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def generate_roc_curve(mu):
    """
    基于正态分布模型生成平滑 ROC 数据
    mu: 分离度 (d-prime)，越大越好
    """
    pfa = np.linspace(0.001, 0.999, 1000)
    pfa = np.concatenate(([0.0], pfa, [1.0]))
    threshold = norm.ppf(pfa)
    pd = norm.cdf(threshold + mu)
    pd[0] = 0.0
    pd[-1] = 1.0
    return pfa, pd

def plot_roc_replica():
    # ================= 参数配置 =================
    # 模拟参考图中的性能层级
    # 1. GCN (参考图中表现最好的黄色线)
    gcn_high_mu = 2.8  # -12dB
    gcn_low_mu  = 2.0  # -14dB
    
    # 2. CNN (参考图中的红色线)
    cnn_high_mu = 1.8  # -12dB
    cnn_low_mu  = 1.2  # -14dB
    
    # 3. MLP (参考图中的蓝色/紫色线，代替 SVM/ANN)
    mlp_high_mu = 1.1  # -12dB
    mlp_low_mu  = 0.6  # -14dB

    # 绘图设置
    plt.figure(figsize=(10, 8))
    
    # 定义 marker 的间隔，防止太密集
    mark_step = 80 
    
    # ================= 绘制 GCN (Proposed) - 黄色/橙色 =================
    # -12dB (实线)
    fpr, tpr = generate_roc_curve(gcn_high_mu)
    plt.plot(fpr, tpr, color='orange', linestyle='-', linewidth=2,
             marker='v', markevery=mark_step, markersize=7, label='GCN-CSS (SNR=-12dB)')
    
    # -14dB (虚线)
    fpr, tpr = generate_roc_curve(gcn_low_mu)
    plt.plot(fpr, tpr, color='orange', linestyle='--', linewidth=2,
             marker='v', markevery=mark_step, markersize=7, mfc='white', label='GCN-CSS (SNR=-14dB)')

    # ================= 绘制 CNN - 红色 =================
    # -12dB (实线)
    fpr, tpr = generate_roc_curve(cnn_high_mu)
    plt.plot(fpr, tpr, color='red', linestyle='-', linewidth=2,
             marker='o', markevery=mark_step, markersize=7, label='CNN (SNR=-12dB)')
    
    # -14dB (虚线)
    fpr, tpr = generate_roc_curve(cnn_low_mu)
    plt.plot(fpr, tpr, color='red', linestyle='--', linewidth=2,
             marker='o', markevery=mark_step, markersize=7, mfc='white', label='CNN (SNR=-14dB)')

    # ================= 绘制 MLP - 蓝色 =================
    # -12dB (实线)
    fpr, tpr = generate_roc_curve(mlp_high_mu)
    plt.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2,
             marker='d', markevery=mark_step, markersize=7, label='MLP (SNR=-12dB)')
    
    # -14dB (虚线)
    fpr, tpr = generate_roc_curve(mlp_low_mu)
    plt.plot(fpr, tpr, color='blue', linestyle='--', linewidth=2,
             marker='d', markevery=mark_step, markersize=7, mfc='white', label='MLP (SNR=-14dB)')

    # ================= 装饰图表 =================
    plt.title('ROC Curves Comparison (Multiple SNRs)', fontsize=16, fontweight='bold')
    plt.xlabel('Probability of false alarm ($P_{fa}$)', fontsize=14)
    plt.ylabel('Probability of detection ($P_d$)', fontsize=14)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    
    # 细致的网格 (参考图风格)
    plt.grid(True, which='major', linestyle='-', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.minorticks_on()
    
    # 图例设置
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    plt.tick_params(axis='both', which='major', labelsize=12)

    # 保存
    save_path = 'roc_comparison_replica.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 复刻版 ROC 对比图已生成: {save_path}")

if __name__ == "__main__":
    plot_roc_replica()