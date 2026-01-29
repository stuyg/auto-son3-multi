import h5py
import numpy as np
import tensorflow as tf
import math
import os

class RadioMLSequence(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size, indices, num_nodes=32, sigma=1.0, mode='binary', num_antennas=1):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.indices = indices
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.mode = mode
        self.num_antennas = num_antennas
        self.num_classes = 2 if mode == 'binary' else 24
        
        # 【优化 1】只保存索引，不再将几十 GB 的数据读入内存
        # 这将极大加快启动速度，并防止内存溢出
        self.total_len = len(self.indices)
        
        print(f"正在初始化生成器 (Antennas={num_antennas})...")
        
        # 预读取少量数据以计算底噪 (只读前 2000 个样本)
        with h5py.File(self.hdf5_path, 'r') as f:
            # 获取特征维度信息
            self.base_feature_dim = f['X'].shape[1] * f['X'].shape[2] // self.num_nodes
            self.feature_dim = self.base_feature_dim * self.num_antennas
            
            # 采样计算底噪
            sample_idx = np.sort(self.indices[:2000])
            temp_X = f['X'][sample_idx]
            temp_Z = f['Z'][sample_idx]
            
            min_snr = np.min(temp_Z)
            noise_idx = np.where(temp_Z == min_snr)[0]
            
            if len(noise_idx) > 0:
                self.noise_std = np.std(temp_X[noise_idx])
            else:
                powers = np.mean(np.var(temp_X, axis=1), axis=1)
                self.noise_std = np.sqrt(np.min(powers))
                
            print(f"✅ 底噪基准计算完毕: Std={self.noise_std:.6f}")

        # 本地索引用于 Shuffle
        self.local_indices = np.arange(len(self.indices))
        np.random.shuffle(self.local_indices)

    def __len__(self):
        return math.ceil(self.total_len / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.total_len)
        current_batch_size = end - start
        
        # 获取当前 Batch 的全局索引
        batch_local_idx = self.local_indices[start:end]
        global_indices = self.indices[batch_local_idx]
        
        # 【优化 2】对索引排序，极大提升 HDF5 读取速度 (随机读 -> 顺序读)
        # 即使打乱了 Batch 内部顺序，对训练梯度下降也没有影响
        sorted_idx = np.sort(global_indices)
        
        # 【优化 3】按需读取硬盘 (On-the-fly Reading)
        with h5py.File(self.hdf5_path, 'r') as f:
            X_batch = f['X'][sorted_idx]
            Z_batch = f['Z'][sorted_idx]
            if self.mode != 'binary':
                Y_batch = f['Y'][sorted_idx]
        
        # 数据处理逻辑...
        if self.mode == 'binary':
            Y_new = np.zeros((current_batch_size, 2), dtype=np.float32)
            Y_new[:, 1] = 1.0 
            
            noise_count = current_batch_size // 2
            if noise_count > 0:
                noise_data = np.random.normal(0, self.noise_std, size=(noise_count, 1024, 2))
                # 覆盖后半部分
                X_batch[-noise_count:] = noise_data
                Y_new[-noise_count:, 0] = 1.0
                Y_new[-noise_count:, 1] = 0.0
                Z_batch[-noise_count:] = -100
            Y_batch = Y_new

        # Reshape & MIMO 扩展
        X_reshaped = X_batch.reshape(-1, self.num_nodes, self.base_feature_dim)
        
        if self.num_antennas > 1:
            X_reshaped = np.tile(X_reshaped, (1, 1, self.num_antennas))
            # 极微小扰动 (防止完全相同的数值导致梯度异常)
            perturbation = np.random.normal(0, 1e-5 * self.noise_std, size=X_reshaped.shape)
            X_reshaped = X_reshaped + perturbation

        X_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)
        
        # 计算邻接矩阵
        diff = tf.expand_dims(X_tensor, 2) - tf.expand_dims(X_tensor, 1)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
        A_batch = tf.exp(-dist_sq / (self.sigma ** 2))
        D = tf.reduce_sum(A_batch, axis=-1, keepdims=True)
        A_batch_norm = A_batch / (D + 1e-6)

        return [X_tensor, A_batch_norm], Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.local_indices)

def get_generators(hdf5_path, batch_size=32, num_nodes=32, split_ratio=0.8, max_samples=None, num_antennas=1):
    # 只读取文件大小，不读取内容
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X'].shape[0]
        
    if max_samples: total_samples = min(total_samples, max_samples)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    
    split_idx = int(total_samples * split_ratio)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    train_gen = RadioMLSequence(hdf5_path, batch_size, train_indices, num_nodes, mode='binary', num_antennas=num_antennas)
    val_gen = RadioMLSequence(hdf5_path, batch_size, val_indices, num_nodes, mode='binary', num_antennas=num_antennas)
    
    return train_gen, val_gen, 2, train_gen.feature_dim