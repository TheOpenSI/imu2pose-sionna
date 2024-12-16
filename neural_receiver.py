import os
import platform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu
from sionna.mimo import StreamManagement
from sionna.channel import CIRDataset, OFDMChannel
from sionna.ofdm import ResourceGrid, ResourceGridMapper, RemoveNulledSubcarriers, LSChannelEstimator, LMMSEEqualizer, ResourceGridDemapper
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from utils.imu_functions import prepare_source_data

os_name = platform.system()
if os_name == 'Linux':
    imu_dataset_path = os.path.expanduser('~/Data/datasets/DIP_IMU_and_Others/') 
else:
    imu_dataset_path = os.path.expanduser('~/datasets/DIP_IMU_and_Others/') 
    
class CustomBinarySource(Layer):
    def __init__(self, num_imu_frames, batch_size, n, quantz_lv=128, dtype=tf.float32, seed=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        self.source_bits, self.source_imu_quantized, self.source_imu_original = prepare_source_data(
            num_imu_frames=num_imu_frames, batch_size=batch_size, n=n, quantization_level=quantz_lv
            )
        self.num_ofdm_rg_batches = self.source_bits.shape[0]

    def call(self, inputs, batch_idx=0):
        # inputs.shape [batch_size, 1, 1, n]
        if isinstance(inputs, int):
            batch_size = inputs
        else:
            batch_size = inputs[0]
        b = self.source_bits[batch_idx]  # [batch_size, 1, 1, n] 
        return b

# Class from https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#BER-Evaluation
class CIRGenerator:
    def __init__(self, a, tau, num_tx, training):
        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]
        self._trainset_size = int(0.8 * self._dataset_size)
        self._testset_size = self._dataset_size - self._trainset_size
        self._num_tx = num_tx
        self._training = training
        # Separate the indices for train and test sets
        self._train_indices = tf.range(self._trainset_size, dtype=tf.int64)
        self._test_indices = tf.range(self._trainset_size, self._dataset_size, dtype=tf.int64)

    def __call__(self):
        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample random users and stack them together
            # Choose the appropriate set of indices
            indices = self._train_indices if self._training else self._test_indices
            dataset_size = self._trainset_size if self._training else self._testset_size
            idx, _, _ = tf.random.uniform_candidate_sampler(
                indices[tf.newaxis, :],
                num_true=dataset_size,
                num_sampled=self._num_tx,
                unique=True,
                range_max=self._dataset_size)
    
            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension  
            # TODO: verify transpose if having more than one TX
            # a = tf.transpose(a, perm=[3, 1, 2, 0, 4, 5, 6])  
            # # print('tau.shape-b: {}'.format(tau.shape))
            # tau = tf.transpose(tau, perm=[2, 1, 0, 3]) 
            # print('tau.shape-a: {}'.format(tau.shape))

            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)
            # print('a.shape: {}'.format(a.shape))
            # print('tau.shape: {}'.format(tau.shape))
            yield a, tau

class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """

    def build(self, input_shape):

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_1 = Conv2D(filters=128,  # params['num_conv_channels']
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=128,  # params['num_conv_channels']
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z

class NeuralReceiver(Layer):
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.

    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.

    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.

    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """

    def build(self, input_shape):
        # Input convolution
        self._input_conv = Conv2D(filters=128,  # params['num_conv_channels']
                                  kernel_size=[3,3],
                                  padding='same',
                                  activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters=2,  # params['num_bits_per_symbol']
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)

    def call(self, inputs):
        y, no = inputs
        # print('y.shape nc: {}'.format(y.shape))

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        # TODO: modify hard defined numbers [batch_size, num_rx_ant, num_ofdm_symbols, num_subcarriers]
        y = tf.ensure_shape(y, [y.shape[0], 16, 14, 128]) 
        y = tf.transpose(y, [0, 2, 3, 1])  # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        # z : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Input conv
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)

        return z

class E2ESystem(Model):
    r"""
    Keras model that implements the end-to-end systems.
    """

    def __init__(self, system, ofdm_params, model_params, a, tau, eval_mode=0, gen_data=True):
        super().__init__()
        self._system = system
        self._eval_mode = eval_mode  # 0, 1, 2: training - ber evaluation - testing (forward)
        
        self._rg = ResourceGrid(
        num_ofdm_symbols=ofdm_params['num_ofdm_symbols'],
        fft_size=ofdm_params['fft_size'],
        subcarrier_spacing=ofdm_params['subcarrier_spacing'],
        num_tx=ofdm_params['num_tx'],
        num_streams_per_tx=ofdm_params['num_tx_ant'],
        cyclic_prefix_length=ofdm_params['cyclic_prefix_length'],
        num_guard_carriers=ofdm_params['num_guard_carriers'],
        dc_null=ofdm_params['dc_null'],
        pilot_pattern=ofdm_params['pilot_pattern'],
        pilot_ofdm_symbol_indices=ofdm_params['pilot_ofdm_symbol_indices']
        )
        
        self._num_tx = ofdm_params['num_tx']
        self._num_tx_ant = ofdm_params['num_tx_ant']
        self._n = int(self._rg.num_data_symbols * ofdm_params['num_bits_per_symbol'])

        if self._eval_mode != 3:
            self._batch_size = model_params['batch_size']
            self._binary_source = BinarySource()
        else:
            # Create data source with quantization:
            # One ofdm symbol with num_ofdm_symbols=14, pilot_ofdm_symbol_indices=[2, 11],
            # fft_size=128, num_guard_carriers=[5,6], dc_null=True, quantization_level=2**8, num_bits_per_symbol=2
            # Thus, n = (128-1-5-6) * (14-2) * 2 = 2784 bits
            self._batch_size = model_params['batch_size']
            self._binary_source = CustomBinarySource(
                num_imu_frames=model_params['num_imu_frames'], batch_size=self._batch_size, 
                n=self._n, quantz_lv=model_params['quantization_level']
                )
        
        if eval_mode == 0 or eval_mode == 1:
            cir_generator = CIRGenerator(a, tau, ofdm_params['num_tx'], training=True)
        else:
            cir_generator = CIRGenerator(a, tau, ofdm_params['num_tx'], training=False)
        # Note that we swap the roles of UE and BS here as we are using uplink
        channel_model = CIRDataset(cir_generator, self._batch_size, ofdm_params['num_rx'], ofdm_params['num_rx_ant'], ofdm_params['num_tx'], ofdm_params['num_tx_ant'],
                                ofdm_params['num_rt_paths'], ofdm_params['num_ofdm_symbols']
                                )
        self._channel = OFDMChannel(channel_model, self._rg, normalize_channel=True, return_channel=True, add_awgn=True)
        del a, tau
        
        # Mapper and stream management
        self._sm = StreamManagement(rx_tx_association=np.ones([1, self._num_tx], bool), num_streams_per_tx=self._num_tx_ant)
        self._mapper = Mapper('qam', ofdm_params['num_bits_per_symbol'])
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._num_bits_per_sym = ofdm_params['num_bits_per_symbol']

        # Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi':  # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(self._rg)
            elif system == 'baseline-ls-estimation':  # LS estimation
                self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
            self._demapper = Demapper("app", "qam", ofdm_params['num_bits_per_symbol'])
        elif system == "neural-receiver":  # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(self._rg, self._sm)

    # @tf.function
    def call(self, batch_size, ebno_db, batch_idx=0):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        # Transmitter
        no = ebnodb2no(ebno_db, self._num_bits_per_sym, 1.0, self._rg)

        if isinstance(self._binary_source, BinarySource):
            # [batch_size, num_tx, num_streams_per_tx, num_data_symbols * num_bits_per_sym]
            b = self._binary_source([batch_size, self._num_tx, self._num_tx_ant, self._n])
        else:
            b = self._binary_source([batch_size, self._num_tx, self._num_tx_ant, self._n], batch_idx=batch_idx)
        # Modulation
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        # Channel
        y, h = self._channel([x_rg, no])

        # Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h)  # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no])  # LS channel estimation with nearest-neighbor
            
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver([y, no])  # [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
            llr = insert_dims(llr, 2, 1)  # Reshape the input to fit what the resource grid demapper is expected (128, 1, 1, 14, 76, 2) -> [batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded  (128, 4, 2, 1536) 
            llr = tf.reshape(llr, [batch_size, self._num_tx, self._num_tx_ant, self._n])  # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._eval_mode == 0 or self._eval_mode == 1:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(b, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Hard decoding
            b_hat = tf.where(llr > 0.0, 1.0, 0.0)
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computatio
        
    def get_binary_source(self):
        return self._binary_source
    
    def get_batch_size(self):
        return self._batch_size
