import h5py
import numpy as np
import tensorflow as tf
import math

class RadioMLSequence(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size, indices, num_nodes=32, sigma=1.0, mode='binary'):
        """
        mode='binary': 强制执行频谱感知任务 (2分类: 噪声 vs 信号)
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.indices = indices
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.mode = mode
        
        # 如果是二分类频谱感知，类别固定为 2
        self.num_classes = 2 if mode == 'binary' else 24
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_len = len(self.indices)
            # 获取原始特征维度
            self.feature_dim = f['X'].shape[1] * f['X'].shape[2] // self.num_nodes

    def __len__(self):
        return math.ceil(self.total_len / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.total_len)
        current_batch_size = end - start
        
        batch_indices = self.indices[start:end]
        sorted_indices = np.sort(batch_indices)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            X_batch = f['X'][sorted_indices] # (Batch, 1024, 2)
            # 我们不需要原来的 Y，因为我们要自己造标签
            # Z (SNR) 仍然有用
            Z_batch = f['Z'][sorted_indices] 

        # =================================================
        # 核心逻辑：如果是频谱感知，我们需要构造 "噪声" 样本
        # =================================================
        if self.mode == 'binary':
            # 1. 创建标签容器 (Batch, 2) -> [Noise, Signal]
            Y_new = np.zeros((current_batch_size, 2), dtype=np.float32)
            Y_new[:, 1] = 1.0 
            
            # 2. 将 Batch 的一半替换为纯噪声
            noise_count = current_batch_size // 2
            
            if noise_count > 0:
                # 【关键修改】计算当前 Batch 信号的平均标准差（幅度）
                # 这样生成的噪声强度就会和信号（或信号里的背景噪声）在同一个量级
                batch_std = np.std(X_batch)
                
                # 使用计算出的 std 生成噪声，而不是默认的 1.0
                noise_data = np.random.normal(0, batch_std, size=(noise_count, X_batch.shape[1], X_batch.shape[2]))
                
                X_batch[-noise_count:] = noise_data
                
                Y_new[-noise_count:, 0] = 1.0
                Y_new[-noise_count:, 1] = 0.0
                
                Z_batch[-noise_count:] = -100

            Y_batch = Y_new
        else:
            # 原始多分类逻辑
            with h5py.File(self.hdf5_path, 'r') as f:
                Y_batch = f['Y'][sorted_indices]

        # =================================================
        # 图构建 (这部分不变)
        # =================================================
        X_reshaped = X_batch.reshape(-1, self.num_nodes, self.feature_dim)
        X_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)
        
        # 动态计算邻接矩阵
        diff = tf.expand_dims(X_tensor, 2) - tf.expand_dims(X_tensor, 1)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
        A_batch = tf.exp(-dist_sq / (self.sigma ** 2))
        D = tf.reduce_sum(A_batch, axis=-1, keepdims=True)
        A_batch_norm = A_batch / (D + 1e-6)

        return [X_tensor, A_batch_norm], Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def get_generators(hdf5_path, batch_size=32, num_nodes=32, split_ratio=0.8, max_samples=None):
    # 这里记得把 mode='binary' 传进去
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X'].shape[0]
    
    if max_samples: total_samples = min(total_samples, max_samples)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    
    split_idx = int(total_samples * split_ratio)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    # 实例化时开启 binary 模式
    train_gen = RadioMLSequence(hdf5_path, batch_size, train_indices, num_nodes, mode='binary')
    val_gen = RadioMLSequence(hdf5_path, batch_size, val_indices, num_nodes, mode='binary')
    
    return train_gen, val_gen, 2, train_gen.feature_dim # 注意这里返回类别数为 2