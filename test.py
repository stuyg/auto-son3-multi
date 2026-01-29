import os
import gc
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, auc

# 导入自定义模型
from model import GCN_CSS, CNN_CSS, MLP_CSS 

# ================= 默认配置 =================
# 这里的路径请根据实际情况修改，如果和 main.py 传参一致也可以
DEFAULT_HDF5_PATH = 'GOLD_XYZ_OSC.0001_1024.hdf5' 
BATCH_SIZE = 32  
NUM_NODES = 32
TARGET_PFA = 0.1 
SAMPLES_PER_SNR = 100 # 每个 SNR 点采样的样本数

# 强制使用 CPU 进行推理 (避免 GPU 显存冲突，且评估数据量不大，CPU 足够)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument('--path', type=str, default=DEFAULT_HDF5_PATH, help='Path to .hdf5 dataset')
    parser.add_argument('--antennas', type=int, default=1, help='Number of antennas M (used during training)')
    parser.add_argument('--model_type', type=str, default='all', choices=['all', 'gcn', 'cnn', 'mlp'], help='Model type to evaluate')
    return parser.parse_args()

# ================= 数据加载 =================
def load_test_data(hdf5_path, samples_per_snr=100, num_antennas=1):
    print(f"🚀 正在加载测试数据 (M={num_antennas}, 每SNR采样={samples_per_snr})...")
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"❌ 找不到数据集: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        Z_all = f['Z'][:]
        unique_snrs = np.unique(Z_all)
        
        selected_indices = []
        np.random.seed(2024)
        
        # 1. 按 SNR 采样
        for snr in unique_snrs:
            indices = np.where(Z_all == snr)[0]
            if len(indices) > samples_per_snr:
                chosen = np.random.choice(indices, samples_per_snr, replace=False)
            else:
                chosen = indices
            selected_indices.extend(chosen)
        selected_indices = np.sort(np.array(selected_indices))
        
        # 2. 读取信号数据 X
        # 为了避免内存爆炸，分块读取
        X_chunks = []
        chunk_size = 2000 
        for i in range(0, len(selected_indices), chunk_size):
            subset = selected_indices[i : i + chunk_size]
            X_chunks.append(f['X'][subset])
        
        X_sig = np.concatenate(X_chunks, axis=0)
        Z_sig = Z_all[selected_indices]
        
        # 3. 估算底噪 (用于生成 H0)
        # 找 -20dB 或最小 SNR 样本
        noise_ref_snr = -20
        noise_idx = np.where(Z_all == noise_ref_snr)[0]
        if len(noise_idx) == 0:
            noise_ref_snr = np.min(Z_all)
            noise_idx = np.where(Z_all == noise_ref_snr)[0]
            
        # 采样计算 Std
        limit = min(2000, len(noise_idx))
        X_floor = f['X'][noise_idx[:limit]]
        noise_std = np.std(X_floor)
        print(f"📉 估计的物理底噪 ({noise_ref_snr}dB): Std={noise_std:.6f}")

    # 4. 生成纯噪声样本 H0
    X_noise = np.random.normal(0, noise_std, size=X_sig.shape).astype(np.float32)
    Z_noise = np.full((len(X_sig), 1), -100.0)
    
    # 合并
    X = np.concatenate([X_noise, X_sig], axis=0)
    Y = np.concatenate([np.zeros(len(X_sig)), np.ones(len(X_sig))])
    Z = np.concatenate([Z_noise, Z_sig])
    
    print(f"✅ 测试数据准备完毕: {X.shape}")
    return X, Y, Z.flatten(), noise_std

# ================= 批处理与多天线模拟 =================
def process_batch(X_raw, num_antennas=1, is_gcn=True):
    # X_raw: (Batch, 1024, 2) -> 需要 reshape 成 (Batch, Nodes, Base_Feats)
    # Base_Feats = 1024*2 / 32 = 64
    base_feat_dim = 1024 * 2 // NUM_NODES
    X_r = X_raw.reshape(-1, NUM_NODES, base_feat_dim)
    
    # 【核心】多天线扩展 (MIMO Simulation)
    # 必须与 dataset.py 中的逻辑保持一致：使用 np.tile 复制
    if num_antennas > 1:
        X_r = np.tile(X_r, (1, 1, num_antennas))
        # 测试时通常不加随机扰动，以保证结果确定性
    
    X_t = tf.convert_to_tensor(X_r, dtype=tf.float32)
    
    if is_gcn:
        # GCN 计算邻接矩阵
        # 使用扩展后的特征计算欧氏距离
        diff = tf.expand_dims(X_t, 2) - tf.expand_dims(X_t, 1)
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        A = tf.exp(-dist) 
        D = tf.reduce_sum(A, axis=-1, keepdims=True)
        A = A / (D + 1e-6)
        return [X_t, A]
    else:
        # CNN/MLP 不需要 A，传个占位符
        batch_size = tf.shape(X_t)[0]
        dummy = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [X_t, dummy]

# ================= 推理主函数 =================
def run_evaluation(model_class, model_path, model_name, X, M):
    print(f"\n🤖 正在评估模型: {model_name} (M={M})...")
    print(f"   权重路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到权重文件 {model_path}")
        return None

    # 清理内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 实例化模型
    # 注意：所有模型类都需要接收 (num_classes, num_nodes)
    model = model_class(2, NUM_NODES)
    
    # Build 模型以加载权重
    # 输入特征维数 = 64 * M
    base_dim = 64
    total_dim = base_dim * M
    
    try:
        # 显式 build，确保形状匹配
        model.build([(None, NUM_NODES, total_dim), (None, NUM_NODES, NUM_NODES)])
        model.load_weights(model_path)
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        print("   提示: 请检查模型结构是否与训练时一致 (model.py)")
        return None

    # 预测循环
    preds = []
    total = len(X)
    is_gcn = 'gcn' in model_name.lower() or 'proposed' in model_name.lower()
    
    for i in range(0, total, BATCH_SIZE):
        bx = X[i : i+BATCH_SIZE]
        inputs = process_batch(bx, num_antennas=M, is_gcn=is_gcn)
        
        # 预测 (返回 Softmax 概率)
        p = model.predict_on_batch(inputs)
        # 取类别 1 (信号存在) 的概率
        preds.append(p[:, 1])
        
        if i % (BATCH_SIZE * 50) == 0:
            print(f"   进度: {i}/{total}", end='\r')
            
    print(f"   进度: {total}/{total} [完成]")
    return np.concatenate(preds)

# ================= 绘图 =================
def plot_results(results_dict, Y_true, Z_snr, M):
    suffix = f"_m{M}"
    
    # 1. Pd vs SNR
    plt.figure(figsize=(10, 6))
    snr_range = np.arange(-20, 31, 2)
    colors = {'GCN': 'red', 'CNN': 'blue', 'MLP': 'green'}
    markers = {'GCN': 'o', 'CNN': 's', 'MLP': '^'}
    
    for name, scores in results_dict.items():
        # 确定样式
        c = 'black'
        m = 'x'
        for k in colors:
            if k in name.upper(): 
                c = colors[k]
                m = markers[k]
                
        # 计算虚警阈值
        noise_scores = scores[Y_true == 0]
        thresh = np.percentile(noise_scores, (1 - TARGET_PFA)*100)
        
        pd_list = []
        for snr in snr_range:
            # 找特定 SNR 的信号样本
            idx = np.where((Y_true == 1) & (np.abs(Z_snr - snr) < 1.0))[0]
            if len(idx) == 0:
                pd_list.append(0)
            else:
                pd = np.mean(scores[idx] > thresh)
                pd_list.append(pd)
        
        plt.plot(snr_range, pd_list, label=name, color=c, marker=m)
        
    plt.title(f'Detection Probability vs SNR (M={M}, Pfa={TARGET_PFA})')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pd')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1.05])
    plt.xlim([-20, 30])
    plt.savefig(f'eval_pd_snr{suffix}.png')
    print(f"📊 图表已保存: eval_pd_snr{suffix}.png")

    # 2. ROC Curve at -10dB
    plt.figure(figsize=(8, 8))
    target_snr = -10
    
    # 筛选 -10dB 附近的信号 + 所有噪声
    sig_idx = np.where((Y_true == 1) & (np.abs(Z_snr - target_snr) < 1.0))[0]
    noise_idx = np.where(Y_true == 0)[0]
    
    if len(sig_idx) > 0:
        y_roc = np.concatenate([np.zeros(len(noise_idx)), np.ones(len(sig_idx))])
        
        for name, scores in results_dict.items():
            s_roc = np.concatenate([scores[noise_idx], scores[sig_idx]])
            fpr, tpr, _ = roc_curve(y_roc, s_roc)
            roc_auc = auc(fpr, tpr)
            
            c = 'black'
            for k in colors:
                if k in name.upper(): c = colors[k]
                
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.4f})", color=c)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve at {target_snr}dB (M={M})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'eval_roc{suffix}.png')
    print(f"📊 图表已保存: eval_roc{suffix}.png")

def main():
    args = parse_args()
    
    # 1. 加载数据
    X, Y, Z, _ = load_test_data(args.path, num_antennas=args.antennas)
    
    # 2. 定义待评估模型
    # 自动根据 M 生成文件名
    models_to_run = []
    
    if args.model_type in ['all', 'gcn']:
        models_to_run.append({
            'name': f'GCN (M={args.antennas})', 
            'class': GCN_CSS, 
            'path': f'best_gcn_m{args.antennas}.h5'
        })
    if args.model_type in ['all', 'cnn']:
        models_to_run.append({
            'name': f'CNN (M={args.antennas})', 
            'class': CNN_CSS, 
            'path': f'best_cnn_m{args.antennas}.h5'
        })
    if args.model_type in ['all', 'mlp']:
        models_to_run.append({
            'name': f'MLP (M={args.antennas})', 
            'class': MLP_CSS, 
            'path': f'best_mlp_m{args.antennas}.h5'
        })

    # 3. 运行评估
    results = {}
    for m in models_to_run:
        scores = run_evaluation(m['class'], m['path'], m['name'], X, args.antennas)
        if scores is not None:
            results[m['name']] = scores
            
    # 4. 绘图
    if len(results) > 0:
        plot_results(results, Y, Z, args.antennas)
    else:
        print("⚠️ 没有得到任何预测结果，无法绘图。")

if __name__ == "__main__":
    main()