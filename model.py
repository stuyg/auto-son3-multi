import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# ==========================================
# 1. 图卷积层 (GraphConv)
# ==========================================
class GraphConv(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.bn = layers.BatchNormalization()

    def build(self, input_shape):
        feature_shape = input_shape[0] 
        input_dim = feature_shape[-1]
        
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        features, adjacency = inputs
        support = tf.matmul(features, self.w)
        output = tf.matmul(adjacency, support) + self.b
        output = self.bn(output)
        
        if self.activation is not None:
            output = self.activation(output)
        return output

# ==========================================
# 2. GCN 模型
# ==========================================
class GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(GCN_CSS, self).__init__()
        self.gcn1 = GraphConv(32, activation='relu')
        self.gcn2 = GraphConv(64, activation='relu')
        self.gcn3 = GraphConv(128, activation='relu')
        
        self.pooling = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        x = self.gcn3([x, a])
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

# ==========================================
# 3. CNN 模型 (修正版)
# ==========================================
class CNN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(CNN_CSS, self).__init__()
        self.conv1 = layers.Conv1D(64, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling1D(2)
        
        self.conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling1D(2)
        
        self.conv3 = layers.Conv1D(128, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        
        self.global_pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs 
        # 输入 x shape: (Batch, Nodes, Features)
        # 例如: (Batch, 32, 64*M)
        
        # 【关键修复】动态 Reshape，支持多天线输入
        # 我们希望恢复为时序信号结构: (Batch, 1024, Channels)
        # 原始单天线: 32 nodes * 64 feats = 2048 numbers = 1024 time * 2 channels
        # M天线: 32 nodes * (64*M) feats = 2048*M numbers = 1024 time * (2*M) channels
        
        batch_size = tf.shape(x)[0]
        total_elements = tf.shape(x)[1] * tf.shape(x)[2] # Nodes * Feats
        
        # 假设时间步长固定为 1024 (物理含义)
        time_steps = 1024
        new_channels = total_elements // time_steps
        
        x = tf.reshape(x, [batch_size, time_steps, new_channels])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

# ==========================================
# 4. MLP 模型
# ==========================================
class MLP_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(MLP_CSS, self).__init__()
        self.flatten = layers.Flatten()
        
        self.dense1 = layers.Dense(1024, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        
        self.dense2 = layers.Dense(512, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        
        self.fc_out = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x, _ = inputs
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        return self.fc_out(x)