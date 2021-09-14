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
    def __init__(self, num_head):
        self.num_attention_head = num_head

    def build(self, input_shape):
        self.W_k = tf.keras.layers.Dense(input_shape[-1])
        self.W_q = tf.keras.layers.Dense(input_shape[-1])
        self.W_v = tf.keras.layers.Dense(input_shape[-1])
        
    def multi(self, vec):
        return tf.reshape(vec, (self.num_attention_head))

    def call(self, query, key, value):
        Wk = self.W_k(key)
        Wq = self.W_q(query)
        Wv = self.W_v(value)
        
        pass