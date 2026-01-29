import tensorflow as tf
import os
import multiprocessing

def train_model(model, train_ds, val_ds, epochs=10, lr=0.001, save_path='best_model.h5'):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path, 
        monitor='val_accuracy', 
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # 自动检测 CPU 核心数
    cpu_count = multiprocessing.cpu_count()
    # 线程数设置：通常设为 CPU 核心数即可
    workers = max(1, cpu_count)
    
    print(f"🚀 开始训练 (Workers: {workers}, Mode: Multithreading)...")
    print(f"💾 权重保存路径: {save_path}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        # 【关键修复】
        # workers > 1: 启用多线程预取数据，解决 IO 瓶颈
        # use_multiprocessing=False: 禁用多进程，防止 CUDA 崩溃
        workers=workers,
        use_multiprocessing=False, 
        max_queue_size=20
    )
    
    return history