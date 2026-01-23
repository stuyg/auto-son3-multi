import tensorflow as tf
import os

def train_model(model, train_ds, val_ds, epochs=10, lr=0.001):
    # 论文公式 (26) 使用 Log Likelihood，等价于 CrossEntropy [cite: 330]
    # 优化器使用 Adam [cite: 336]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # 回调函数：保存最佳模型
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_gcn_model.h5', 
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

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
    return history