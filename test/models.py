import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sionna.utils import log10, insert_dims, ebnodb2no, flatten_last_dims


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
    

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
    

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=False) 
    self.pos_encoding = positional_encoding(length=vocab_size, depth=d_model)
    self.dense_layer = tf.keras.layers.Dense(d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    # print('x input shape: {}'.format(x.shape))  # (batch_size, seq_len)
    length = x.shape[1]
    # z = self.embedding(x)
    # z = tf.expand_dims(x, axis=-1)
    # z = self.dense_layer(z)
    # print('z pos shape: {}'.format(z.shape))
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :self.d_model]
    return x


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    z = self.add([x, self.seq(x)])
    z = self.layer_norm(z) 
    return z


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    z = self.self_attention(x)
    z = self.ffn(z)
    return z


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.conv = tf.keras.layers.Conv2D(filters=self.d_model, kernel_size=(3, 3), padding='same', activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.dense_layer = tf.keras.layers.Dense(d_model, activation='relu')

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.final_layer = tf.keras.layers.Conv2D(
      filters=target_vocab_size,  # params['num_bits_per_symbol'] 
      kernel_size=[3,3], 
      padding='same', 
      activation=None
      )

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len, d_model)
    # print('x input shape: {}'.format(x.shape))
    z = self.conv(x)
    # print('z conv shape: {}'.format(z.shape))
    z = self.flatten(z)
    # print('z flat shape: {}'.format(z.shape))
    # z = self.dense_layer(z)
    z = tf.reshape(z, (z.shape[0], 14*128, 33))
    # print('z dense shape: {}'.format(z.shape))
    z = self.pos_embedding(z)  # Shape `(batch_size, seq_len, d_model)`.
    # print('z post shape: {}'.format(x.shape))
    
    # Add dropout.
    z = self.dropout(z)

    for i in range(self.num_layers):
      z = self.enc_layers[i](z)
      
    # Final linear layer output
    # print('z enc_layer: {}'.format(z.shape))
    z = tf.reshape(z, (z.shape[0], 14, 128, z.shape[-1]))  # [batch size, num time samples, num subcarriers, num_channels]
    z = self.final_layer(z)  # shape `(batch_size, target_len, d_model)`

    try:
      del z._keras_mask
    except AttributeError:
      pass

    return z  # Shape `(batch_size, seq_len, target_vocab_size)`.
  

class TranformerReceiver(tf.keras.layers.Layer):
  '''
  Neural receiver with attention layers
  
  Input
  ------
  y: [batch_size, num_rx_antennas, num_ofdm_symbols, num_subcarriers], tf.complex
  
  no: [batch_size], tf.float32
  
  Output
  ------
  soft_x: [batch_size, num_ofdm_symbols, num_subcarriers, num_bits_per_symbols]
  '''
  
  def build(self, input_shape):
    self._y_shape = input_shape[0]
    self._encoder = Encoder(
      num_layers=4, 
      d_model=33, 
      num_heads=3, 
      dff=132, 
      vocab_size=self._y_shape[2] * self._y_shape[3],  # (self._y_shape[1]*2+1) * self._y_shape[2] * self._y_shape[3]
      target_vocab_size=2
      )
    
  def call(self, inputs):
    y, no = inputs
    no = log10(no)
    # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
    y = tf.ensure_shape(y, self._y_shape) 
    y = tf.transpose(y, [0, 2, 3, 1])  # Putting antenna dimension last [batch_size, num_ofdm_symbols, num_subcarriers, num_rx_antennas]
    # print('no.shape: {}'.format(no.shape))
    no = insert_dims(no, 3, 1)
    no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
    # z : [batch_size, num_ofdm_symbols, num_subcarriers, 2*num_rx_antenna + 1]
    z = tf.concat([tf.math.real(y), tf.math.imag(y), no], axis=-1)
    # print('z.shape: {}'.format(z.shape))
    z_shape = z.shape
    # reshape z into the input_shape of the Transformer [batch_size, seq_len, d_model], 
    # z = tf.reshape(z, [z_shape[0], z_shape[1]*z_shape[2], z_shape[3]])
    # print('z_new.shape: {}'.format(z.shape))
    z = self._encoder(z)  # shape [bach_size, seq_len, 2]
    z = tf.reshape(z, [z_shape[0], z_shape[1], z_shape[2], z.shape[-1]])
    
    return z
    
def test_encoder():
  sample_encoder = Encoder(num_layers=4, d_model=512, num_heads=4, dff=2048, vocab_size=1000, target_vocab_size=10)
  pt = tf.random.uniform([6, 10], minval=0, maxval=1000)
  print('pt: {}'.format(pt))
  output = sample_encoder(pt, training=False)
  print('output: {}'.format(output))
  print('pt.shape: {}'.format(pt.shape))
  print('output.shape: {}'.format(output.shape))
  
def test_transfomer_receiver():
  y_real = tf.random.uniform(shape=(32, 16, 14, 128), minval=-1.0, maxval=1.0, dtype=tf.float32)
  y_img = tf.random.uniform(shape=(32, 16, 14, 128), minval=-1.0, maxval=1.0, dtype=tf.float32)
  y = tf.complex(y_real, y_img)
  print('y.shape: {}/{}'.format(y.shape, y.dtype))
  no = tf.random.uniform(shape=[32], minval=0.0, maxval=1.0, dtype=tf.float32)
  transfomer_receiver = TranformerReceiver()
  output = transfomer_receiver([y, no])
  print('output: {}'.format(output))
  # print('output.shape: {}'.format(output.shape))
    

if __name__ == '__main__':
    # test_encoder()
    test_transfomer_receiver()

    
