import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

# 引入你的模型定义
from model import GCN_CSS, ANN_CSS, CNN_CSS

# ================= 配置区域 =================
# 1. 精确匹配截图中的 SNR 范围
SNR_LIST = [-18, -16, -14, -12, -10, -8] 

# 2. 采样设置
SAMPLES_PER_SNR = 1000   # 每个SNR点评估的样本数
SKLEARN_TRAIN_SAMPLES = 20000  # 训练SVM/KMeans用的样本数

# 3. 颜色方案 (匹配截图)
COLORS = {
    'GCN-CSS': 'green',
    'CNN': 'blue',
    'ANN': 'cyan',
    'SVM': 'darkred',
    'KMeans': 'orange'
}
# ===========================================

def load_keras_model(model_class, weights_path, num_nodes, num_features):
    """实例化 Keras 模型并加载权重"""
    model = model_class(num_classes=2, num_nodes=num_nodes)
    try:
        # Build模型以初始化权重形状
        model.build([(None, num_nodes, num_features), (None, num_nodes, num_nodes)])
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ [Keras] 成功加载: {weights_path}")
            return model
        else:
            print(f"⚠️ [Keras] 缺失权重: {weights_path} (将跳过此模型或输出随机结果)")
            return None
    except Exception as e:
        print(f"❌ 加载模型出错 {weights_path}: {e}")
        return None

def train_sklearn_models(hdf5_path, num_nodes):
    """快速训练 SVM 和 KMeans 用于对比"""
    print("\n⚙️ 正在训练基准模型 (SVM & KMeans)...")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 读取一部分数据
        X = f['X'][:SKLEARN_TRAIN_SAMPLES]
        half = len(X) // 2
        
        # 构造训练集：一半噪声，一半信号
        X_signal = X[:half]
        batch_std = np.std(X_signal)
        X_noise = np.random.normal(0, batch_std, size=X_signal.shape)
        
        X_train = np.concatenate([X_noise, X_signal], axis=0)
        y_train = np.concatenate([np.zeros(half), np.ones(half)]) # 0:Noise, 1:Signal
        
        # Flatten (N, 32, Feat) -> (N, -1)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # SVM
    print("   -> Training SVM (LinearSVC)...")
    svm = LinearSVC(max_iter=1000, dual=False)
    svm.fit(X_train_flat, y_train)
    
    # KMeans
    print("   -> Training KMeans (Unsupervised)...")
    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=256, n_init=3, random_state=42)
    kmeans.fit(X_train_flat)
    
    # 自动纠正 KMeans 的标签方向 (Cluster ID 是随机的)
    sample_preds = kmeans.predict(X_train_flat[half:]) # 预测信号部分
    # 如果大部分信号被分成了 0，说明 Cluster 0 是信号，需要反转
    invert_kmeans = np.mean(sample_preds) < 0.5 
    
    return svm, kmeans, invert_kmeans

def get_eval_data(f, snr, num_samples, num_nodes):
    """获取特定 SNR 的测试数据"""
    all_snrs = f['Z'][:]
    indices = np.where(np.abs(all_snrs - snr) < 0.5)[0]
    
    if len(indices) == 0:
        return None, None, None
    
    selected = np.random.choice(indices, min(len(indices), num_samples), replace=False)
    X_signal = f['X'][selected]
    
    # 生成噪声
    batch_std = np.std(X_signal)
    X_noise = np.random.normal(0, batch_std, size=X_signal.shape)
    
    # 合并
    X_combined = np.concatenate([X_noise, X_signal], axis=0)
    Y_combined = np.concatenate([np.zeros(len(X_noise)), np.ones(len(X_signal))])
    
    # Keras 输入格式
    feat_dim = X_combined.shape[1] * X_combined.shape[2] // num_nodes
    X_reshaped = X_combined.reshape(-1, num_nodes, feat_dim)
    X_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)
    
    # 动态邻接矩阵计算
    diff = tf.expand_dims(X_tensor, 2) - tf.expand_dims(X_tensor, 1)
    dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
    A_val = tf.exp(-dist_sq) # sigma=1.0
    
    # Sklearn 输入格式
    X_flat = X_combined.reshape(X_combined.shape[0], -1)
    
    return [X_tensor, A_val], X_flat, Y_combined

def plot_reproduction(results, snr_list):
    """绘制与截图一致的柱状图"""
    print("\n🎨 正在绘图...")
    
    # 确保顺序: GCN, CNN, ANN, SVM, KMeans
    ordered_keys = ['GCN-CSS', 'CNN', 'ANN', 'SVM', 'KMeans']
    
    snrs = [str(s) for s in snr_list]
    x = np.arange(len(snrs))
    width = 0.15 # 柱子宽度
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制每一组柱子
    for i, model_name in enumerate(ordered_keys):
        if model_name not in results: continue
        
        scores = np.array(results[model_name]) * 100 # 转为百分比
        
        # 计算偏移量，使柱子居中
        offset = (i - len(ordered_keys)/2 + 0.5) * width
        
        ax.bar(x + offset, scores, width, 
               label=model_name, 
               color=COLORS.get(model_name, 'gray'),
               edgecolor='white', linewidth=0.5)

    # 设置样式
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_title('Classification Accuracy vs SNR', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(snrs)
    
    # Y轴范围 0 - 100
    ax.set_ylim([0, 100])
    
    # 图例
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 保存
    save_path = 'reproduced_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存为: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='HDF5 dataset path')
    parser.add_argument('--nodes', type=int, default=32)
    args = parser.parse_args()
    
    # 1. 准备深度学习模型
    keras_configs = {
        'GCN-CSS': {'class': GCN_CSS, 'file': 'best_gcn_model.h5'},
        'CNN':     {'class': CNN_CSS, 'file': 'best_cnn_model.h5'},
        'ANN':     {'class': ANN_CSS, 'file': 'best_ann_model.h5'},
    }
    
    # 自动推断特征维度
    with h5py.File(args.path, 'r') as f:
        sample_x = f['X'][0]
        num_feat = sample_x.shape[0] * sample_x.shape[1] // args.nodes
    
    loaded_models = {}
    for name, cfg in keras_configs.items():
        m = load_keras_model(cfg['class'], cfg['file'], args.nodes, num_feat)
        if m: loaded_models[name] = m

    # 2. 训练 Sklearn 模型
    svm, kmeans, km_invert = train_sklearn_models(args.path, args.nodes)
    
    # 3. 收集结果
    results = {k: [] for k in ['GCN-CSS', 'CNN', 'ANN', 'SVM', 'KMeans']}
    
    print("\n📊 开始评估...")
    with h5py.File(args.path, 'r') as f:
        for snr in SNR_LIST:
            print(f" -> Testing SNR = {snr} dB", end='\r')
            
            k_in, sk_in, y_true = get_eval_data(f, snr, SAMPLES_PER_SNR, args.nodes)
            
            if k_in is None:
                for k in results: results[k].append(0.5) # 默认猜测
                continue
                
            # 评估 DL 模型
            for name, model in loaded_models.items():
                pred = np.argmax(model.predict(k_in, verbose=0), axis=1)
                results[name].append(accuracy_score(y_true, pred))
            
            # 评估 SVM
            results['SVM'].append(accuracy_score(y_true, svm.predict(sk_in)))
            
            # 评估 KMeans
            km_pred = kmeans.predict(sk_in)
            if km_invert: km_pred = 1 - km_pred
            results['KMeans'].append(accuracy_score(y_true, km_pred))
            
    # 4. 绘图
    plot_reproduction(results, SNR_LIST)

if __name__ == "__main__":
    main()