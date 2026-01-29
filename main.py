import os
import argparse
import tensorflow as tf
from dataset import get_generators 
from model import GCN_CSS, CNN_CSS, MLP_CSS 
from training import train_model

# 显存配置
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ [GPU] 已检测到 {len(gpus)} 个 GPU，显存动态增长已开启。")
    except RuntimeError as e:
        print(f"❌ 显存设置失败: {e}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU 运行。")

def main():
    parser = argparse.ArgumentParser(description="GCN/CNN/MLP Spectrum Sensing MIMO Experiment")
    parser.add_argument('--path', type=str, required=True, help='Path to .hdf5 dataset')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'cnn', 'mlp'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--samples', type=int, default=None)
    # 【新增】天线数量参数
    parser.add_argument('--antennas', type=int, default=1, help='Number of antennas M (e.g., 1, 2, 4, 6, 8)')
    
    args = parser.parse_args()
    
    print(f"🚀 实验配置: Nodes={args.nodes}, Antennas(M)={args.antennas}...")
    
    # 获取生成器
    train_gen, val_gen, num_classes, num_features = get_generators(
        hdf5_path=args.path,
        batch_size=args.batch_size,
        num_nodes=args.nodes,
        split_ratio=0.8,
        max_samples=args.samples,
        num_antennas=args.antennas # 传入天线参数
    )
    
    print(f"✅ 数据准备完毕。输入特征维数: {num_features} (Base_Dim * {args.antennas})")
    
    # 选择模型并设置保存路径
    save_name = f"best_{args.model_type}_m{args.antennas}.h5"
    
    if args.model_type == 'gcn':
        print("构建 GCN 模型...")
        model = GCN_CSS(num_classes=num_classes, num_nodes=args.nodes)
    elif args.model_type == 'cnn':
        print("构建 CNN 模型...")
        model = CNN_CSS(num_classes=num_classes, num_nodes=args.nodes)
    elif args.model_type == 'mlp':
        print("构建 MLP 模型...")
        model = MLP_CSS(num_classes=num_classes, num_nodes=args.nodes)
    
    # Build 模型
    # 输入形状: [(Batch, Nodes, Feats), (Batch, Nodes, Nodes)]
    model.build([(None, args.nodes, num_features), (None, args.nodes, args.nodes)])
    model.summary()
    
    # 开始训练
    train_model(model, train_gen, val_gen, epochs=args.epochs, lr=args.lr, save_path=save_name)

if __name__ == "__main__":
    main()