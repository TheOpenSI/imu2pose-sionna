"""
fc_receiver.py
--------------
Fully-connected (FC-only) neural receiver baseline for the OFDM stage.

Purpose (Reviewer 5, comments R5.1 / R5.4 / R5.6):
    The proposed neural receiver (neural_receiver.py::NeuralReceiver) is a faithful
    re-implementation of the residual-CONVOLUTIONAL receiver of Ait Aoudia & Hoydis [23]
    and DeepRx [24]. To isolate the benefit of the residual-convolutional design --
    i.e., exploiting time-frequency correlations across the OFDM resource grid -- this
    module provides an ablation that is identical in depth, width, normalisation and
    residual structure, but uses 1x1 kernels instead of 3x3 kernels.

    A 1x1 kernel processes every resource element (OFDM symbol x subcarrier) INDEPENDENTLY,
    using only that element's per-antenna observation. It therefore implements exactly the
    "baseline that treats each subcarrier independently" requested by the reviewer, and is
    the natural fully-connected (position-wise MLP) counterpart of the convolutional receiver.
    The ONLY ablated variable is the spatial (time-frequency) receptive field.

Integration: see INTEGRATION.md. In short, this layer is a drop-in replacement for
NeuralReceiver and produces an identical output shape, so E2ESystem only needs a new
branch that instantiates FCNeuralReceiver under the system name 'neural-receiver-fc'.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from sionna.utils import log10, insert_dims


class FCResidualBlock(Layer):
    r"""
    Position-wise residual block: identical to neural_receiver.ResidualBlock but with
    1x1 kernels. Layer normalisation and the skip connection are retained so that the
    only difference from the convolutional block is the absence of spatial mixing.

    Input / Output : [batch, num_ofdm_symbols, num_subcarriers, num_conv_channels], tf.float
    """

    def __init__(self, num_channels=128, **kwargs):
        super().__init__(**kwargs)
        self._num_channels = num_channels

    def build(self, input_shape):
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=self._num_channels, kernel_size=[1, 1],
                              padding='same', activation=None)
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=self._num_channels, kernel_size=[1, 1],
                              padding='same', activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z)
        # Skip connection
        z = z + inputs
        return z


class FCNeuralReceiver(Layer):
    r"""
    Fully-connected (per-resource-element) neural receiver.

    Mirrors neural_receiver.NeuralReceiver exactly -- same input handling, same number
    of blocks (4), same width (128), same LayerNorm, same residual skips -- but every
    convolution uses a 1x1 kernel. As a result the network has NO access to neighbouring
    OFDM symbols or subcarriers: each resource element is decoded from its own 16-antenna
    observation alone. This isolates the contribution of the 3x3 spatial convolutions in
    the proposed receiver.

    Input
    ------
    y  : [batch, num_rx_ant, num_ofdm_symbols, num_subcarriers], tf.complex
    no : [batch], tf.float32

    Output
    -------
    : [batch, num_ofdm_symbols, num_subcarriers, num_bits_per_symbol], tf.float (LLRs)
    """

    def __init__(self, num_channels=128, num_blocks=4, num_bits_per_symbol=2,
                 num_rx_ant=16, num_ofdm_symbols=14, num_subcarriers=128, **kwargs):
        super().__init__(**kwargs)
        self._num_channels = num_channels
        self._num_blocks = num_blocks
        self._num_bits_per_symbol = num_bits_per_symbol
        self._num_rx_ant = num_rx_ant
        self._num_ofdm_symbols = num_ofdm_symbols
        self._num_subcarriers = num_subcarriers

    def build(self, input_shape):
        # Input 1x1 conv (position-wise dense over the 2*num_rx_ant + 1 input channels)
        self._input_conv = Conv2D(filters=self._num_channels, kernel_size=[1, 1],
                                  padding='same', activation=None)
        # Residual blocks (1x1)
        self._res_blocks = [FCResidualBlock(self._num_channels)
                            for _ in range(self._num_blocks)]
        # Output 1x1 conv -> LLRs (num_bits_per_symbol filters)
        self._output_conv = Conv2D(filters=self._num_bits_per_symbol, kernel_size=[1, 1],
                                   padding='same', activation=None)

    def call(self, inputs):
        y, no = inputs

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Match NeuralReceiver's input handling exactly.
        y = tf.ensure_shape(
            y, [y.shape[0], self._num_rx_ant, self._num_ofdm_symbols, self._num_subcarriers])
        y = tf.transpose(y, [0, 2, 3, 1])  # antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        # z : [batch, num ofdm symbols, num subcarriers, 2*num_rx_ant + 1]
        z = tf.concat([tf.math.real(y), tf.math.imag(y), no], axis=-1)

        z = self._input_conv(z)
        for blk in self._res_blocks:
            z = blk(z)
        z = self._output_conv(z)
        return z
