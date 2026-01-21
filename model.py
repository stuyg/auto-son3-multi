import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class GraphConv(layers.Layer):
    """
    自定义图卷积层
    Implementation of GCN layer: Output = Activation(A * X * W)
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape 是一个列表: [(batch, nodes, feats), (batch, nodes, nodes)]
        # feature_shape: (batch, nodes, input_dim)
        feature_shape = input_shape[0] 
        input_dim = feature_shape[-1]
        
        # 权重矩阵 W: (input_dim, units) [cite: 229]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        
        # 偏置 b
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        # inputs = [X, A]
        # X: (Batch, Nodes, Feats)
        # A: (Batch, Nodes, Nodes)
        features, adjacency = inputs
        
        # 1. 特征变换 (Feature Transformation): Z = XW
        # (Batch, Nodes, Input_Dim) @ (Input_Dim, Units) -> (Batch, Nodes, Units)
        support = tf.matmul(features, self.w)
        
        # 2. 消息传递 (Message Passing): Output = AZ
        # (Batch, Nodes, Nodes) @ (Batch, Nodes, Units) -> (Batch, Nodes, Units)
        output = tf.matmul(adjacency, support)
        
        output = output + self.b
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class GCN_CSS(Model):
    def __init__(self, num_classes, num_nodes):
        super(GCN_CSS, self).__init__()
        
        # 架构参考论文 Table II 
        # Layer 1: GCN (32 filters) + ReLU
        self.gcn1 = GraphConv(32, activation='relu')
        
        # Layer 2: GCN (64 filters) + ReLU
        self.gcn2 = GraphConv(64, activation='relu')
        
        # Layer 3: GCN (128 filters) + ReLU
        self.gcn3 = GraphConv(128, activation='relu')
        
        # Pooling: Global Sum Pooling
        self.pooling = layers.GlobalMaxPooling1D() # 或者 GlobalAveragePooling1D, 论文使用的是 Sum/Max 聚合
        
        # FC Layers
        # FC 1
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        
        # FC 2 (Output) [cite: 296]
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # inputs 包含两个张量: [features, adjacency]
        x, a = inputs
        
        # GCN Block
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        x = self.gcn3([x, a])
        
        # Pooling: (Batch, Nodes, 128) -> (Batch, 128)
        # 论文提到 "Graph pooling... coarsens the graph" [cite: 284]
        x = self.pooling(x)
        
        # Classification
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)