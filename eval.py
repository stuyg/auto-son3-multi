import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from model import GCN_CSS 

# ================= 配置区域 =================
HDF5_PATH = '/root/autodl-tmp/SS/GNN_CSS/GOLD_XYZ_OSC.0001_1024.hdf5'
MODEL_WEIGHTS_PATH = 'best_gcn_model.h5'

BATCH_SIZE = 128
NUM_NODES = 32
# 【关键修改】这里必须是 2
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
    all_true_classes = []
    all_snrs = []
    
    # 既然是二分类，我们在评估时也需要模拟出“纯噪声”数据
    # 因为 HDF5 原始文件里只有信号。
    # 这里我们采用简单的策略：只评估 HDF5 里的数据，视为“信号(Class 1)”
    # 为了画 ROC，我们需要自己在内存里生成噪声数据(Class 0)
    
    num_batches = int(np.ceil(total_test_samples / BATCH_SIZE))
    
    with h5py.File(hdf5_path, 'r') as f:
        X_dataset = f['X']
        Z_dataset = f['Z']
        
        for i in range(num_batches):
            batch_start = start_idx + i * BATCH_SIZE
            batch_end = min(start_idx + (i + 1) * BATCH_SIZE, start_idx + total_test_samples)
            if batch_start >= batch_end: break
            
            # --- 1. 读取真实信号 (Class 1) ---
            X_signal = X_dataset[batch_start:batch_end]
            Z_signal = Z_dataset[batch_start:batch_end]
            current_batch_len = X_signal.shape[0]
            
            # --- 2. 生成纯噪声 (Class 0) ---
            # 为了保持平衡，生成同样数量的噪声
            X_noise = np.random.normal(0, 1.0, size=X_signal.shape).astype(np.float32)
            Z_noise = np.full((current_batch_len, 1), -100.0) # 噪声SNR标记为 -100
            
            # --- 3. 合并 ---
            X_combined = np.concatenate([X_noise, X_signal], axis=0)
            # 标签: 0=噪声, 1=信号
            Y_combined = np.concatenate([np.zeros(current_batch_len), np.ones(current_batch_len)])
            Z_combined = np.concatenate([Z_noise, Z_signal])
            
            # --- 4. 预测 ---
            inputs = process_batch(X_combined, NUM_NODES, SIGMA)
            preds = model.predict_on_batch(inputs) # (2*Batch, 2)
            
            # --- 5. 存储 ---
            all_pred_probs.append(preds)
            all_true_classes.append(Y_combined) # 这里已经是 0/1 整数了
            all_snrs.append(Z_combined)
            
            if i % 10 == 0:
                print(f"进度: {i}/{num_batches} batches processed...", end='\r')

    print("\n✅ 评估完成，正在合并结果...")
    y_probs = np.concatenate(all_pred_probs)
    y_true = np.concatenate(all_true_classes)
    snrs = np.concatenate(all_snrs)
    y_pred_class = np.argmax(y_probs, axis=1)
    
    return y_true, y_pred_class, y_probs, snrs

def plot_results(y_true, y_pred, y_probs, snrs):
    # --- 1. ROC Curve (二分类核心) ---
    plt.figure(figsize=(8, 8))
    
    # 取出属于 Class 1 (信号) 的概率
    y_score = y_probs[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Alarm Rate (P_fa)')
    plt.ylabel('Detection Probability (P_d)')
    plt.title('Spectrum Sensing ROC (Noise vs Signal)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve_binary.png')
    print("已保存: roc_curve_binary.png")
    
    # --- 2. Accuracy vs SNR (只看信号部分的检测率) ---
    # 我们只关心真实标签为 1 (信号) 的样本在不同 SNR 下被预测对的概率 (Pd)
    snrs = snrs.flatten()
    # 过滤掉我们手动生成的噪声(SNR=-100)
    signal_indices = np.where(snrs > -99)[0]
    
    if len(signal_indices) > 0:
        signal_snrs = snrs[signal_indices]
        signal_true = y_true[signal_indices]
        signal_pred = y_pred[signal_indices]
        
        unique_snrs = np.sort(np.unique(signal_snrs))
        pd_scores = []
        
        print("\n========== 检测概率 (Pd) vs SNR ==========")
        for snr in unique_snrs:
            idx = np.where(signal_snrs == snr)[0]
            if len(idx) == 0: continue
            # 对于信号样本，准确率就是检测概率 (Pd)
            pd = accuracy_score(signal_true[idx], signal_pred[idx])
            pd_scores.append(pd)
            
        plt.figure(figsize=(10, 6))
        plt.plot(unique_snrs, pd_scores, 'r-o', linewidth=2, label='Detection Prob (Pd)')
        plt.title('Detection Probability vs SNR')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Probability of Detection (Pd)')
        plt.grid(True)
        plt.savefig('pd_vs_snr.png')
        print("已保存: pd_vs_snr.png")

    # --- 3. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
    plt.title('Confusion Matrix (Binary)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_binary.png')
    print("已保存: confusion_matrix_binary.png")

if __name__ == "__main__":
    # 1. 加载模型 (必须是 2 分类模型)
    print("正在加载二分类模型...")
    model = GCN_CSS(num_classes=NUM_CLASSES, num_nodes=NUM_NODES)
    feat_dim = 1024 * 2 // NUM_NODES
    # 这里的 Build 形状
    model.build([(None, NUM_NODES, feat_dim), (None, NUM_NODES, NUM_NODES)])
    
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
    except ValueError as e:
        print("\n❌ 错误: 权重加载失败！")
        print("可能原因: 你的 best_gcn_model.h5 是按 24 类训练的，但现在代码是 2 类。")
        print("解决方法: 请先运行 main.py 重新训练二分类模型。")
        exit()
    
    # 2. 获取测试集
    test_count, start_index = get_total_test_samples(HDF5_PATH)
    
    # 3. 运行评估
    # 注意：我们在 evaluate 函数内部生成了一半的噪声数据，所以实际评估样本量会翻倍
    y_true, y_pred, y_probs, snrs = evaluate_entire_dataset(model, HDF5_PATH, start_index, test_count)
    
    # 4. 绘图
    plot_results(y_true, y_pred, y_probs, snrs)