import tensorflow as tf
import os
import sys

print("="*40)
print(f"Python Version: {sys.version.split()[0]}")
print(f"TensorFlow Version: {tf.__version__}")
print("="*40)

# 1. 检查 TF 是基于哪个 CUDA 版本编译的
try:
    build_info = tf.sysconfig.get_build_info()
    print(f"TensorFlow Built With:")
    print(f" - CUDA:  {build_info.get('cuda_version', 'Unknown')}")
    print(f" - cuDNN: {build_info.get('cudnn_version', 'Unknown')}")
except Exception as e:
    print(f"无法获取构建信息: {e}")

print("-" * 40)

# 2. 检查系统实际安装的 CUDA 版本 (通过 nvcc)
print("System Actual CUDA (nvcc):")
exit_code = os.system('nvcc --version | grep release')
if exit_code != 0:
    print("❌ 未找到 nvcc 命令 (可能未安装 CUDA Toolkit 或 路径没配)")

print("-" * 40)

# 3. 检查能否检测到 GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"Detected GPUs: {len(gpus)}")
if len(gpus) > 0:
    print("✅ TensorFlow 能看到 GPU")
    try:
        # 尝试进行一次简单的计算，触发 cuDNN 初始化
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        print("✅ 简单的矩阵乘法成功 (CUDA 库加载正常)")
    except Exception as e:
        print(f"❌ 计算失败 (典型的版本不兼容错误):\n{e}")
else:
    print("❌ TensorFlow 看不到 GPU (通常是 CUDA 版本太低或未正确安装)")
print("="*40)