import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset import load_ideal_pu_signals, generate_h0_h1
from model import build_gcn_css
from dataloader import create_dataloader
from metrics import predict_by_snr, plot_pd_pf  # 新增导入

# -------------------------- 强制CPU训练 --------------------------
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.optimizer.set_jit(False)

# -------------------------- 训练+测试逻辑 --------------------------
def train_gcn_css(data_path: str, target_pf: float = 0.1):
    """
    训练模型 + 测试「固定Pf下」不同SNR的Pd
    :param data_path: H5数据文件路径
    :param target_pf: 目标固定虚警概率（默认0.1）
    """
    # 1. 加载PU信号
    ideal_pu = load_ideal_pu_signals(
        data_path,
        min_snr=10,
        max_samples=500
    )
    
    # 2. 生成训练数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_h0_h1(
        ideal_pu,
        total_samples=1000,
        target_snr_range=(-18, 10)
    )
    print(f"📊 数据划分：训练集{len(X_train)} | 验证集{len(X_val)} | 测试集{len(X_test)}")

    # 3. 创建数据加载器
    batch_size = 2
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # 4. 构建+训练模型
    model = build_gcn_css()
    model.summary()

    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint("best_gcn_css_cpu.h5", save_best_only=True, monitor='val_loss')
    ]

    print("🚀 开始CPU训练...")
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=False,
        workers=1
    )

    # 5. 基础测试集评估
    print("📈 基础测试集评估...")
    test_loss, test_acc = model.evaluate(
        test_loader,
        verbose=1,
        use_multiprocessing=False,
        workers=1
    )
    print(f"✅ 基础测试结果：损失={test_loss:.4f} | 准确率={test_acc:.4f}")

    # 6. 不同SNR下的Pd/Pf测试（核心修改：传入target_pf）
    print(f"\n============= 开始测试不同SNR下的Pd (固定Pf={target_pf:.2f}) =============")
    # 定义要测试的SNR列表（覆盖低/中/高SNR）
    test_snr_list = [-18, -12, -6, 0, 6, 10]
    # 计算「固定Pf下」的Pd
    snr_results = predict_by_snr(
        model=model,
        ideal_pu=ideal_pu,
        snr_list=test_snr_list,
        target_pf=target_pf,  # 传入固定Pf值
        n_samples_per_snr=100  # 每个SNR测试100个H0+100个H1样本
    )
    
    # 7. 打印汇总结果（突出固定Pf）
    print(f"\n============= 固定Pf={target_pf:.2f} 下Pd/Pf汇总结果 =============")
    for snr in sorted(snr_results.keys()):
        pd, actual_pf = snr_results[snr]
        print(f"SNR={snr}dB: Pd={pd:.4f}, 实际Pf={actual_pf:.4f} (目标Pf={target_pf:.2f})")
    
    # 8. 绘制「固定Pf下」的Pd-SNR曲线（传入target_pf）
    plot_pd_pf(
        snr_results,
        target_pf=target_pf,
        save_path=f"pd_pf_curve_fixed_pf_{target_pf:.2f}.png"  # 文件名标注固定Pf值
    )

    return model, snr_results

if __name__ == "__main__":
    DATA_PATH = "/root/autodl-tmp/SS/GOLD_XYZ_OSC.0001_1024.hdf5"
    # 可自定义固定Pf值（如0.05/0.1/0.2）
    TARGET_PF = 0.1
    model, snr_results = train_gcn_css(DATA_PATH, target_pf=TARGET_PF)