import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from model import GCN_CSS 

# ================= 配置区域 =================
HDF5_PATH = '/root/autodl-tmp/radioml2018/GCN_CSS/GOLD_XYZ_OSC.0001_1024.hdf5'
MODEL_WEIGHTS_PATH = 'best_gcn_model.h5'

BATCH_SIZE = 128
NUM_NODES = 32
NUM_CLASSES = 2 
SIGMA = 1.0

# 显存设置
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# ===========================================

def get_total_test_samples(hdf5_path, split_ratio=0.8):
    with h5py.File(hdf5_path, 'r') as f:
        total_len = f['X'].shape[0]
        start_idx = int(total_len * split_ratio)
        return total_len - start_idx, start_idx

def process_batch(X_raw, num_nodes=32, sigma=1.0):
    feature_dim = X_raw.shape[1] * X_raw.shape[2] // num_nodes
    X_reshaped = X_raw.reshape(-1, num_nodes, feature_dim)
    X_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)

    diff = tf.expand_dims(X_tensor, 2) - tf.expand_dims(X_tensor, 1)
    dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
    A_batch = tf.exp(-dist_sq / (sigma ** 2))
    
    D = tf.reduce_sum(A_batch, axis=-1, keepdims=True)
    A_norm = A_batch / (D + 1e-6)
    
    return [X_tensor, A_norm]

def evaluate_entire_dataset(model, hdf5_path, start_idx, total_test_samples):
    print(f"🚀 开始二分类全量评估 (Spectrum Sensing)...")
    
    all_pred_probs = [] 
    all_true_labels = [] 
    all_snrs = []
    
    num_batches = int(np.ceil(total_test_samples / BATCH_SIZE))
    
    with h5py.File(hdf5_path, 'r') as f:
        X_dataset = f['X']
        Z_dataset = f['Z']
        
        for i in range(num_batches):
            batch_start = start_idx + i * BATCH_SIZE
            batch_end = min(start_idx + (i + 1) * BATCH_SIZE, start_idx + total_test_samples)
            if batch_start >= batch_end: break
            
            # 1. 真实信号
            X_signal = X_dataset[batch_start:batch_end]
            Z_signal = Z_dataset[batch_start:batch_end]
            current_batch_len = X_signal.shape[0]
            
            # 2. 生成噪声 (动态匹配功率)
            batch_std = np.std(X_signal)
            X_noise = np.random.normal(0, batch_std, size=X_signal.shape).astype(np.float32)
            Z_noise = np.full((current_batch_len, 1), -100.0) 
            
            # 3. 合并
            X_combined = np.concatenate([X_noise, X_signal], axis=0)
            Y_combined = np.concatenate([np.zeros(current_batch_len), np.ones(current_batch_len)])
            Z_combined = np.concatenate([Z_noise, Z_signal])
            
            # 4. 预测
            inputs = process_batch(X_combined, NUM_NODES, SIGMA)
            preds = model.predict_on_batch(inputs) 
            
            all_pred_probs.append(preds[:, 1]) 
            all_true_labels.append(Y_combined)
            all_snrs.append(Z_combined)
            
            if i % 10 == 0:
                print(f"进度: {i}/{num_batches} batches processed...", end='\r')

    print("\n✅ 评估完成，正在合并结果...")
    y_scores = np.concatenate(all_pred_probs)
    y_true = np.concatenate(all_true_labels)
    snrs = np.concatenate(all_snrs).flatten()
    
    return y_true, y_scores, snrs

def plot_results(y_true, y_scores, snrs, target_pfa=0.1):
    # 字体设置 (可选)
    plt.rcParams.update({'font.size': 12})
    
    # ==========================================
    # 1. 绘制 Multi-SNR ROC Curve (最重要的新图)
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # 选取几个典型的 SNR 进行展示
    target_snrs_to_plot = [-20, -15, -10, -5, 0, 10]
    colors = ['gray', 'purple', 'red', 'orange', 'green', 'blue']
    
    # 提取所有噪声样本（作为公共的负样本集）
    noise_indices = np.where(y_true == 0)[0]
    noise_scores = y_scores[noise_indices]
    
    print("\n========== 各信噪比 ROC 分析 ==========")
    for snr, color in zip(target_snrs_to_plot, colors):
        # 提取该 SNR 下的信号样本
        # 使用 np.isclose 防止浮点数精度问题
        signal_indices = np.where((y_true == 1) & (np.abs(snrs - snr) < 0.5))[0]
        
        if len(signal_indices) == 0:
            print(f"⚠️ 数据集中未找到 SNR={snr}dB 的样本，跳过。")
            continue
            
        signal_scores_local = y_scores[signal_indices]
        
        # 构造临时的标签和分数：(全量噪声 + 当前SNR信号)
        y_true_local = np.concatenate([np.zeros(len(noise_scores)), np.ones(len(signal_scores_local))])
        y_scores_local = np.concatenate([noise_scores, signal_scores_local])
        
        fpr, tpr, _ = roc_curve(y_true_local, y_scores_local)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, label=f'SNR={snr}dB (AUC={roc_auc:.3f})')
        print(f"SNR={snr}dB: AUC={roc_auc:.4f}")

    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Alarm Rate ($P_{fa}$)')
    plt.ylabel('Detection Probability ($P_d$)')
    plt.title('ROC Curves at Different SNRs')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('roc_curve_multi_snr.png', dpi=300)
    print("✅ 已保存: roc_curve_multi_snr.png (包含不同SNR的平滑ROC)")

    # ==========================================
    # 2. 绘制 Pd vs SNR
    # ==========================================
    noise_scores_sorted = np.sort(noise_scores)
    threshold_idx = int((1 - target_pfa) * len(noise_scores))
    threshold = noise_scores_sorted[threshold_idx]
    
    unique_snrs = np.sort(np.unique(snrs))
    unique_snrs = unique_snrs[unique_snrs > -50] # 过滤占位符
    
    pd_list = []
    for snr in unique_snrs:
        idx = np.where((y_true == 1) & (np.abs(snrs - snr) < 0.1))[0]
        if len(idx) == 0: continue
        pd = np.mean(y_scores[idx] > threshold)
        pd_list.append(pd)

    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, pd_list, 'r-o', linewidth=2, label=f'GCN-CSS ($P_{{fa}}$={target_pfa})')
    plt.title(f'Detection Probability vs SNR ($P_{{fa}}$={target_pfa})')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.ylim([0.0, 1.0])
    plt.xlim([np.min(unique_snrs), np.max(unique_snrs)])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('pd_vs_snr.png', dpi=300)
    print("✅ 已保存: pd_vs_snr.png")

if __name__ == "__main__":
    # 1. 加载模型
    model = GCN_CSS(num_classes=NUM_CLASSES, num_nodes=NUM_NODES)
    feat_dim = 1024 * 2 // NUM_NODES
    model.build([(None, NUM_NODES, feat_dim), (None, NUM_NODES, NUM_NODES)])
    
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        exit()
    
    # 2. 评估
    test_count, start_index = get_total_test_samples(HDF5_PATH)
    y_true, y_scores, snrs = evaluate_entire_dataset(model, HDF5_PATH, start_index, test_count)
    
    # 3. 绘图
    plot_results(y_true, y_scores, snrs, target_pfa=0.1)