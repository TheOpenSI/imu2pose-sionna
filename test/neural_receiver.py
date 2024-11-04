import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization
from tensorflow.nn import relu

from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import (ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer,
                         RemoveNulledSubcarriers, ResourceGridDemapper)
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
sionna.config.seed = 42

# Code based on https://nvlabs.github.io/sionna/examples/Neural_Receiver.html

def init_config():
    # Channel configuration
    system_params = {
        'carrier_frequency': 3.5e9,  # Hz
        'delay_spread': 100e-9,  # s
        'cdl_model': 'C',  # CDL model to use
        'speed': 10.0,  # Speed for evaluation and training [m/s]
        'ebno_db_min': -5.0,  # SNR range for evaluation and training [dB]
        'ebno_db_max': 10.0,
        'subcarrier_spacing': 30e3,  # Hz - OFDM waveform configuration
        'fft_size': 128,  # No of subcarriers in the resource grid, including the null-subcarrier and the guard bands
        'num_ofdm_symbols': 14,  # Number of OFDM symbols forming the resource grid
        'dc_null': True,  # Null the DC subcarrier
        'num_guard_carriers': [5, 6],  # Number of guard carriers on each side
        'pilot_pattern': "kronecker",  # Pilot pattern
        'pilot_ofdm_symbol_indices': [2, 11],  # Index of OFDM symbols carrying pilots
        'cyclic_prefix_length': 0,  # Simulation in frequency domain. This is useless
        'num_bits_per_symbol': 2,  # Modulation and coding configuration
        'coderate': 0.5  # Coderate for LDPC code

    }

    # Neural receiver configuration
    nn_params = {
        'num_conv_channels': 128,   # Number of convo channels for the convo layers forming the neural receiver
        'num_training_iterations': 30000,  # Number of training iterations
        'training_batch_size': 128,  # Training batch size
        'model_weights_path': 'data/neural_receiver_weights',
        'results_filename': 'data/neural_receiver_results'
    }

    return system_params, nn_params

def link_config(params):
    stream_manager = StreamManagement(np.array([[1]]),  # Receiver-transmitter association matrix
                                      1)  # One stream per transmitter
    resource_grid = ResourceGrid(num_ofdm_symbols=params['num_ofdm_symbols'],
                                 fft_size=params['fft_size'],
                                 subcarrier_spacing=params['subcarrier_spacing'],
                                 num_tx=1,
                                 num_streams_per_tx=1,
                                 cyclic_prefix_length=params['cyclic_prefix_length'],
                                 dc_null=params['dc_null'],
                                 pilot_pattern=params['pilot_pattern'],
                                 pilot_ofdm_symbol_indices=params['pilot_ofdm_symbol_indices'],
                                 num_guard_carriers=params['num_guard_carriers']
                                 )

    # Codeword length
    n = int(resource_grid.num_data_symbols * params['num_bits_per_symbol'])
    # Number of information bits per codeword
    k = int(n * params['coderate'])

    ut_antenna = Antenna(polarization="single",
                         polarization_type="V",
                         antenna_pattern="38.901",
                         carrier_frequency=params['carrier_frequency'])

    bs_array = AntennaArray(num_rows=1,
                            num_cols=1,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=params['carrier_frequency'])

    ## Transmitter
    binary_source = BinarySource()
    mapper = Mapper("qam", params['num_bits_per_symbol'])
    rg_mapper = ResourceGridMapper(resource_grid)

    ## Channel
    cdl = CDL(params['cdl_model'], params['delay_spread'], params['carrier_frequency'],
              ut_antenna, bs_array, "uplink", min_speed=params['speed'])
    channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

    ## Receiver
    neural_receiver = NeuralReceiver()
    rg_demapper = ResourceGridDemapper(resource_grid, stream_manager)  # Used to extract data-carrying resource elements

    return (ut_antenna, bs_array, binary_source, n, stream_manager, resource_grid, mapper, rg_mapper,
            channel, neural_receiver, rg_demapper)

def step_forward(params, bin_source, n_bits, mapper, rg_mapper, channel, neural_receiver):
    # Perform one forward step through the end-to-end system
    batch_size = 64
    ebno_db = tf.fill([batch_size], 5.0)
    no = ebnodb2no(ebno_db, params['num_bits_per_symbol'], params['coderate'])

    # Generate codewords
    c = bin_source([batch_size, 1, 1, n_bits])
    print("c shape: ", c.shape)
    # Map bits to QAM symbols
    x = mapper(c)
    print("x shape: ", x.shape)
    # Map the QAM symbols to a resource grid
    x_rg = rg_mapper(x)
    print("x_rg shape: ", x_rg.shape)

    ######################################
    ## Channel
    # A batch of new channel realizations is sampled and applied at every inference
    no_ = expand_to_rank(no, tf.rank(x_rg))
    y, _ = channel([x_rg, no_])
    print("y shape: ", y.shape)

    ######################################
    ## Receiver
    # The neural receiver computes LLRs from the frequency domain received symbols and N0
    y = tf.squeeze(y, axis=1)
    llr = neural_receiver([y, no])
    print("llr shape: ", llr.shape)
    # Reshape the input to fit what the resource grid demapper is expected
    llr = insert_dims(llr, 2, 1)
    # Extract data-carrying resource elements. The other LLRs are discarded
    llr = rg_demapper(llr)
    llr = tf.reshape(llr, [batch_size, 1, 1, n])
    print("Post RG-demapper LLRs: ", llr.shape)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
    bce = tf.reduce_mean(bce)
    rate = tf.constant(1.0, tf.float32) - bce / tf.math.log(2.)
    print(f"Rate: {rate:.2E} bit")


def train_neural_receiver(ofdm_params, n_bits, mapper, resource_grid, channel, ut_ant, bs_ant, stream_manager):
    # Range of SNRs over which the systems are evaluated
    ebno_dbs = np.arange(ofdm_params['ebno_db_min'],  # Min SNR for evaluation
                         ofdm_params['ebno_db_max'],  # Max SNR for evaluation
                         1.0)  # Step

    model = E2ESystem(system='neural-receiver', ofdm_params=ofdm_params, n_bits=n_bits,
                      mapper=mapper, resource_grid=resource_grid, channel=channel,
                      ut_ant=ut_ant, bs_ant=bs_ant, stream_manager=stream_manager, training=True
                      )

    # Sampling a batch of SNRs
    ebno_db = tf.random.uniform(shape=[], minval=ofdm_params['ebno_db_min'], maxval=ofdm_params['ebno_db_max'])
    # optimizer = tf.keras.optimizers.Adam() # for Ubuntu
    optimizer = tf.keras.optimizers.legacy.Adam()  # for Mac
    num_training_iterations = 1000

    for i in range(num_training_iterations):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[], minval=ofdm_params['ebno_db_min'], maxval=ofdm_params['ebno_db_max'])
        # Forward pass
        with tf.GradientTape() as tape:
            rate = model(128, ebno_db)
            # Tensorflow optimizers only know how to minimize loss function.
            # Therefore, a loss function is defined as the additive inverse of the BMD rate
            loss = -rate
        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Periodically printing the progress
        if i % 10 == 0:
            print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_training_iterations, rate.numpy()), end='\r')

        # Save the weights in a file
    weights = model.get_weights()
    model_weights_path = 'data/neural_receiver_weights'
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)

def test_neural_receiver(ofdm_params, n_bits, mapper, resource_grid, channel, ut_ant, bs_ant, stream_manager):
    model = E2ESystem(system='neural-receiver', ofdm_params=ofdm_params, n_bits=n_bits,
                      mapper=mapper, resource_grid=resource_grid, channel=channel,
                      ut_ant=ut_ant, bs_ant=bs_ant, stream_manager=stream_manager
                      )

    # Run one inference to build the layers and loading the weights
    model(1, tf.constant(10.0, tf.float32))
    model_weights_path = 'data/neural_receiver_weights'
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    # Evaluations
    BLER = {}
    # Range of SNRs over which the systems are evaluated
    ebno_dbs = np.arange(ofdm_params['ebno_db_min'],  # Min SNR for evaluation
                         ofdm_params['ebno_db_max'],  # Max SNR for evaluation
                         1.0)  # Step
    _, bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
    BLER['neural-receiver'] = bler.numpy()

    model = E2ESystem('baseline-ls-estimation', ofdm_params=ofdm_params, n_bits=n_bits,
                      mapper=mapper, resource_grid=resource_grid, channel=channel,
                      ut_ant=ut_ant, bs_ant=bs_ant, stream_manager=stream_manager
                      )
    _, bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
    BLER['baseline-ls-estimation'] = bler.numpy()

    model = E2ESystem('baseline-perfect-csi', ofdm_params=ofdm_params, n_bits=n_bits,
                      mapper=mapper, resource_grid=resource_grid, channel=channel,
                      ut_ant=ut_ant, bs_ant=bs_ant, stream_manager=stream_manager
                      )
    _, bler = sim_ber(model, ebno_dbs, batch_size=128, num_target_block_errors=100, max_mc_iter=100)
    BLER['baseline-perfect-csi'] = bler.numpy()

    plt.figure(figsize=(10, 6))
    # Baseline - Perfect CSI
    plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
    # Baseline - LS Estimation
    plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
    # Neural receiver
    plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c=f'C2', label=f'Neural receiver')
    #
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim((1e-4, 1.0))
    plt.legend()
    plt.tight_layout()
    plt.show()


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

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
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

    As the three considered end-to-end systems (perfect CSI baseline, LS estimation baseline, and neural receiver) share most of
    the link components (transmitter, channel model, outer code...), they are implemented using the same Keras model.

    When instantiating the Keras model, the parameter ``system`` is used to specify the system to setup,
    and the parameter ``training`` is used to specified if the system is instantiated to be trained or to be evaluated.
    The ``training`` parameter is only relevant when the neural

    At each call of this model:
    * A batch of codewords is randomly sampled, modulated, and mapped to resource grids to form the channel inputs
    * A batch of channel realizations is randomly sampled and applied to the channel inputs
    * The receiver is executed on the post-DFT received samples to compute LLRs on the coded bits.
      Which receiver is executed (baseline with perfect CSI knowledge, baseline with LS estimation, or neural receiver) depends
      on the specified ``system`` parameter.
    * If not training, the outer decoder is applied to reconstruct the information bits
    * If training, the BMD rate is estimated over the batch from the LLRs and the transmitted bits

    Parameters
    -----------
    system : str
        Specify the receiver to use. Should be one of 'baseline-perfect-csi', 'baseline-ls-estimation' or 'neural-receiver'

    training : bool
        Set to `True` if the system is instantiated to be trained. Set to `False` otherwise. Defaults to `False`.
        If the system is instantiated to be trained, the outer encoder and decoder are not instantiated as they are not required for training.
        This significantly reduces the computational complexity of training.
        If training, the bit-metric decoding (BMD) rate is computed from the transmitted bits and the LLRs. The BMD rate is known to be
        an achievable information rate for BICM systems, and therefore training of the neural receiver aims at maximizing this rate.

    Input
    ------
    batch_size : int
        Batch size

    no : scalar or [batch_size], tf.float
        Noise variance.
        At training, a different noise variance should be sampled for each batch example.

    Output
    -------
    If ``training`` is set to `True`, then the output is a single scalar, which is an estimation of the BMD rate computed over the batch. It
    should be used as objective for training.
    If ``training`` is set to `False`, the transmitted information bits and their reconstruction on the receiver side are returned to
    compute the block/bit error rate.
    """

    def __init__(self, system, ofdm_params, n_bits, mapper,
                 resource_grid, channel, ut_ant, bs_ant, stream_manager, training=False):
        super().__init__()
        self._system = system
        self._training = training

        self._n = int(resource_grid.num_data_symbols * ofdm_params['num_bits_per_symbol'])
        self._k = int(n * ofdm_params['coderate'])
        self._num_bits_per_sym = ofdm_params['num_bits_per_symbol']
        self._coderate = ofdm_params['coderate']

        # Transmitter
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", ofdm_params['num_bits_per_symbol'])
        self._rg_mapper = ResourceGridMapper(resource_grid)

        # Channel
        # A 3GPP CDL channel model is used
        cdl = CDL(ofdm_params['cdl_model'], ofdm_params['delay_spread'], ofdm_params['carrier_frequency'],
                  ut_ant, bs_ant, "uplink", min_speed=ofdm_params['speed'])
        self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True)

        # Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi':  # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation':  # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager, )
            self._demapper = Demapper("app", "qam", ofdm_params['num_bits_per_symbol'])
        elif system == "neural-receiver":  # Neural receiver
            self._neural_receiver = NeuralReceiver()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager)
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        if not training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    @tf.function
    def call(self, batch_size, ebno_db):
        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)

        # Transmitter
        no = ebnodb2no(ebno_db, self._num_bits_per_sym, self._coderate)
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, 1, 1, self._n])
        else:
            b = self._binary_source([batch_size, 1, 1, self._k])
            c = self._encoder(b)
        # Modulation
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        # Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no_])

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
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1)  # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, n])  # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            bce = tf.nn.sigmoid_cross_entropy_with_logits(c, llr)
            bce = tf.reduce_mean(bce)
            rate = tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            return rate
        else:
            # Outer decoding
            b_hat = self._decoder(llr)
            return b, b_hat  # Ground truth and reconstructed information bits returned for BER/BLER computatio


if __name__ == '__main__':
    system_params, nn_params = init_config()
    (ut_ant, bs_ant, binary_source, n, sm, rg, mapper, rg_mapper,
     channel, neural_receiver, rg_demapper) = link_config(system_params)
    step_forward(system_params, binary_source, n, mapper, rg_mapper, channel, neural_receiver)
    train_neural_receiver(system_params, n, mapper, rg, channel, ut_ant, bs_ant, sm)
    test_neural_receiver(system_params, n, mapper, rg, channel, ut_ant, bs_ant, sm)

