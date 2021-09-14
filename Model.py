import numpy as np
import os
import tensorflow as tf


class MySelfAttentionLayer1D (tf.keras.layers.Layer):
    def __init__(self):
        super(MySelfAttentionLayer1D, self).__init__()
    
    def call(self, query, key, value):

        QK = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = QK / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights

class MyMultiHeadAttentionLayer (tf.keras.layers.Layer):
    def __init__(self, model_dim, num_head):
        super(MyMultiHeadAttentionLayer, self).__init__()
        self.num_attention_head = num_head
        self.model_dim = model_dim
        self.depth = self.model_dim // self.num_attention_head

    def build(self, input_shape):
        self.W_k = tf.keras.layers.Dense(input_shape[-1])
        self.W_q = tf.keras.layers.Dense(input_shape[-1])
        self.W_v = tf.keras.layers.Dense(input_shape[-1])
        self.final_layer = tf.keras.layers.Dense(input_shape[-1])

        self.self_attn = MySelfAttentionLayer1D()
    
    def multi(self, vec, batch_size):
        # slice last dimension of size model_dim into (num_attention_head x depth) dimentions
        vec = tf.reshape(vec, (batch_size, -1, self.num_attention_head, self.depth))
        return tf.transpose(vec, perm=[0, 2, 1, 3])

    def call(self, query, key, value):
        batch_size =  tf.shape(query)[0]

        Wq = self.multi(self.W_q(query), batch_size)
        Wk = self.multi(self.W_k(key), batch_size)
        Wv = self.multi(self.W_v(value), batch_size)

        attn_output, attn_weights = self.self_attn(Wq, Wk, Wv)

        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat_attn_output = tf.reshape(attn_output, (batch_size, -1, self.model_dim))
        output = self.final_layer(concat_attn_output)
        return output, attn_weights

class MyEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads):
        super(MyEncoder, self).__init__()
        self.num_heads = num_heads

    def build(self, input_shape):
        self.MHA = MyMultiHeadAttentionLayer(input_shape[-1], self.num_heads)
        self.final_layer = tf.keras.layers.Dense(input_shape[-1])

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, input):
        x1 = input
        x2, _ = self.MHA(input, input, input)

        #normalize
        x3 = self.layer_norm_1(x1 + x2)
        x4 = self.final_layer(x3)

        #normalize
        x5 = self.layer_norm_2(x3 + x4)
        return x5 


