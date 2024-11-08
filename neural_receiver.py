import os
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
from sionna.rt import PlanarArray, Receiver, Transmitter
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from utils.imu_functions import pre_processing_imu, imu_to_binary, binary_to_imu, numpy_to_tensorflow_source

import matplotlib
matplotlib.use('QtAgg')


def prepare_source_data(shape, quantization_level, down_sample=1):
    # load IMU data
    path = os.path.expanduser('~/datasets/DIP_IMU_and_Others/processed_test.npz')
    data = np.load(path)['imu']
    data = np.squeeze(data)
    # Downsample original data for visualization
    data = data[::down_sample]
    print('Original IMU shape: {}'.format(data.shape))
    data = pre_processing_imu(data)

    # Quantization
    bits_per_value = int(np.ceil(np.log2(quantization_level)))
    source_bits = imu_to_binary(data, quantization_level)
    print('Source bits len: {}'.format(len(source_bits)))
    batch_len = int(np.prod(shape))
    batch_bits = source_bits[:batch_len]
    print('Batch bits len: {}'.format(len(batch_bits)))
    imu_shape = (shape[0], shape[1], shape[2], int(shape[3]/quantization_level))
    imu = binary_to_imu(batch_bits, quantization_level, imu_shape, -1.0, 1.0)
    b = numpy_to_tensorflow_source(batch_bits, shape)
    print('imu.shape: {}'.format(imu.shape))

    return b, imu


def generate_channel_impulse_responses(scene, num_cirs, batch_size_cir, rg, num_tx_ant, num_paths, uplink=True, save=True):
    max_depth = 5
    min_gain_db = -130  # in dB / ignore any position with less than -130 dB path gain
    max_gain_db = 0  # in dB / ignore any position with more than 0 dB path gain
    # Sample points within a 10-400m radius around the transmitter
    min_dist = 10  # in m
    max_dist = 400  # in m

    if save:
        # Update coverage map
        print('Update coverage map ...')
        cm = scene.coverage_map(max_depth=max_depth, diffraction=True, cm_cell_size=(1.0, 1.0), combining_vec=None,
                                precoding_vec=None, num_samples=int(1e6)
                                )
        
        # remove current RX (user) and then simulating multiple random-positinoed RXs (users) later with ray tracing
        scene.remove("rx")  

        # Configure antenna array for all transmitters (=UEs)
        scene.rx_array = PlanarArray(num_rows=1, num_cols=int(num_tx_ant/2), vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='iso', polarization='cross'
                                 )
        # TODO: update orientation and height of the rx_array based on the IMU attached on the user head
        # Create batch_size receivers
        # sample batch_size random user positions from coverage map
        print('Update user positions ... ')
        ue_pos, _ = cm.sample_positions(num_pos=batch_size_cir,
                                        metric="path_gain",
                                        min_val_db=min_gain_db,
                                        max_val_db=max_gain_db,
                                        min_dist=min_dist,
                                        max_dist=max_dist)
        ue_pos = tf.squeeze(ue_pos)
        for i in range(batch_size_cir):
            rx = Receiver(name=f"rx-{i}",
                          position=ue_pos[i],  # Random position sampled from coverage map
                          )
            scene.add(rx)

        print('Generating batches of CIRs (a, tau): ...')
        a, tau = None, None
        num_runs = int(np.ceil(num_cirs / batch_size_cir))

        # loop for creating random batch of a and tau
        for idx in range(num_runs):
            print('Progress: {}/{}'.format(idx, num_runs), end='\r')
            # Sample random user positions
            ue_pos, _ = cm.sample_positions(
                num_pos=batch_size_cir,
                metric="path_gain",
                min_val_db=min_gain_db,
                max_val_db=max_gain_db,
                min_dist=min_dist,
                max_dist=max_dist
            )
            ue_pos = tf.squeeze(ue_pos)

            # Update all receiver positions
            for idx in range(batch_size_cir):
                scene.receivers[f"rx-{idx}"].position = ue_pos[idx]

            # Simulate CIR
            paths = scene.compute_paths(
                max_depth=max_depth,
                diffraction=True,
                num_samples=1e6
            )  # shared between all tx in a scene

            # Transform paths into channel impulse responses
            paths.reverse_direction = True  # Convert to uplink direction
            paths.apply_doppler(sampling_frequency=rg.subcarrier_spacing,
                                num_time_steps=rg.num_ofdm_symbols,
                                tx_velocities=[0.0, 0.0, 0.0],
                                rx_velocities=[3.0, 3.0, 0.0])

            # We fix here the maximum number of paths to 75 which ensures
            # that we can simply concatenate different channel impulse reponses
            a_, tau_ = paths.cir(num_paths=num_paths)
            del paths  # Free memory

            if a is None:
                a = a_.numpy()
                tau = tau_.numpy()
            else:
                # Concatenate along the num_tx dimension
                a = np.concatenate([a, a_], axis=3)
                tau = np.concatenate([tau, tau_], axis=2)

        # Exchange the num_tx and batchsize dimensions
        a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
        tau = np.transpose(tau, [2, 1, 0, 3])

        # Remove CIRs that have no active link (i.e., a is all-zero)
        p_link = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5, 6))
        a = a[p_link > 0., ...]
        tau = tau[p_link > 0., ...]

        np.save('data/a_dataset.npy', a)
        np.save('data/tau_dataset.npy', tau)
    else:
        a = np.load('data/a_dataset.npy')
        tau = np.load('data/tau_dataset.npy')

    return a, tau

# Class from https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#BER-Evaluation
class CIRGenerator:
    def __init__(self,
                 a,
                 tau,
                 num_tx,
                 reverse_direction):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx
        self._reverse_direction = reverse_direction

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample random users and stack them together
            idx, _, _ = tf.random.uniform_candidate_sampler(
                tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                num_true=self._dataset_size,
                num_sampled=self._num_tx,
                unique=True,
                range_max=self._dataset_size)

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            a = tf.transpose(a, [3, 1, 2, 0, 4, 5, 6])
            tau = tf.transpose(tau, [2, 1, 0, 3])

            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)
            yield a, tau

class CustomBinarySource(Layer):
    def __init__(self, shape=(10200, 1, 1, 1536), quantz_lv=128, dtype=tf.float32, seed=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        self.shape = shape
        self.mini_batch_size = int(204 * quantz_lv / 1536)
        self.b_all, self.imu_all = prepare_source_data(shape=shape, quantization_level=quantz_lv)
        # b.shape and imu.shape conversion: [17, 1, 1, 1536] ~ [1, 204]
        print('b_all.shape: {}'.format(self.b_all.shape))  # [10200, 1, 1, 1536]
        print('imu_all.shape: {}'.format(self.imu_all.shape))  # [600, 204]  # why (10200, 1, 1, 12)?


    def call(self, inputs, batch_idx=0):
        # inputs.shape [170, 1, 1, 1536]
        # Obtain batch size from inputs shape or as a direct parameter
        if isinstance(inputs, int):
            batch_size = inputs
        else:
            batch_size = inputs[0]
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        # Ensure indices are within bounds of `b_all`
        if end_idx > self.b_all.shape[0]:
            raise ValueError(f"Batch index out of range: requested {end_idx} but only {self.b_all.shape[0]} samples available.")
        b = tf.gather(self.b_all, tf.range(start_idx, end_idx))  # [170, 1, 1, 1536] ~ (10, 204)
        return b

    def get_batch_imu_shape(self, batch_size):
        imu_shape = [int(batch_size / self.mini_batch_size), 204]
        return imu_shape

    def get_dataset_len(self):
        return self.shape[0]

    def b_to_imu(self, b):
        '''
        Convert binary batch `b` into `imu` data
        '''
        # b [batch_size, 1, 1, n]

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

    def __init__(self, system, ofdm_params, model_params, scene, eval_mode=0, gen_data=True):
        super().__init__()
        self._system = system
        self._eval_mode = eval_mode  # 0, 1, 2: training - ber evaluation - testing (forward)
        
        self._rg = ResourceGrid(
        num_ofdm_symbols=ofdm_params['num_ofdm_symbols'],
        fft_size=ofdm_params['fft_size'],
        subcarrier_spacing=ofdm_params['subcarrier_spacing'],
        cyclic_prefix_length=ofdm_params['cyclic_prefix_length'],
        num_guard_carriers=ofdm_params['num_guard_carriers'],
        dc_null=ofdm_params['dc_null'],
        pilot_pattern=ofdm_params['pilot_pattern'],
        pilot_ofdm_symbol_indices=ofdm_params['pilot_ofdm_symbol_indices']
        )
        
        self._n = int(self._rg.num_data_symbols * ofdm_params['num_bits_per_symbol'])
        self._num_batches = model_params['num_batches']
        self._quantization_level = model_params['quantization_level']
        self._imu_frame_per_batch_tx = model_params['imu_frame_per_batch_tx']
        
        # Create data source with quantization:
        # One ofdm symbol with num_ofdm_symbols=14, pilot_ofdm_symbol_indices=[2, 11],
        # fft_size=76, num_guard_carriers=[5,6], dc_null=True, quantization_level=2**7, num_bits_per_symbol=2
        # can transmit n = (76-1-5-6) * (14-2) * 2 = 1536 bits ~ 12 IMU features
        # Thus, one frame of 204 IMU features is equivalent to a batch transmission with shape [mini_batch_size, 1, 1, 1536] ~ imu_shape [1, 204]
        # In other words, we need mini_batch_size=17 OFDM frames (17 ms) to transmit one IMU frame.
        # That's why be set batch_size = mini_batch_size * imu_frame_per_batch_tx
        if self._eval_mode != 3:
            self._batch_size = model_params['batch_size']
        else:
            mini_batch_size = int(204 * self._quantization_level / self._n)
            self._batch_size = mini_batch_size * self._imu_frame_per_batch_tx
        
        if self._eval_mode != 3:
            self._binary_source = BinarySource()
        else:
            self._binary_source = CustomBinarySource(shape=(self._batch_size * self._num_batches, 1, 1, self._n), quantz_lv=self._quantization_level)

        # Customized channel
        num_paths = 14
        a, tau = generate_channel_impulse_responses(scene, 5000, 200, self._rg, ofdm_params['num_tx_ant'], num_paths, True, gen_data)
        print('a.shape: {}'.format(a.shape))
        print('tau.shape: {}'.format(tau.shape))
        cir_generator = CIRGenerator(a, tau, ofdm_params['num_tx'], True)
        # Note that we swap the roles of UE and BS here as we are using uplink
        channel_model = CIRDataset(cir_generator, self._batch_size, ofdm_params['num_rx'], ofdm_params['num_rx_ant'], ofdm_params['num_tx'], ofdm_params['num_tx_ant'],
                                num_paths, ofdm_params['num_ofdm_symbols']
                                )
        self._channel = OFDMChannel(channel_model, self._rg, normalize_channel=True, return_channel=True, add_awgn=True)
        del a, tau

        # Mapper and stream management
        self._sm = StreamManagement(rx_tx_association=np.ones([1, 1], bool), num_streams_per_tx=1)
        self._mapper = Mapper('qam', ofdm_params['num_bits_per_symbol'])
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._num_bits_per_sym = ofdm_params['num_bits_per_symbol']

        # Receiver
        # Three options for the receiver depending on the value of `system`
        if "baseline" in system:
            if system == 'baseline-perfect-csi':  # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(self._rg)
                self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
            elif system == 'baseline-ls-estimation':  # LS estimation
                self._removed_null_subc = RemoveNulledSubcarriers(self._rg)
                self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
            self._demapper = Demapper("maxlog", "qam", ofdm_params['num_bits_per_symbol'])
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
            b = self._binary_source([batch_size, 1, 1, self._n])
        else:
            b = self._binary_source([batch_size, 1, 1, self._n], batch_idx=batch_idx)
        # Modulation
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        # Channel
        # A batch of new channel realizations is sampled and applied at every inference
        # no_ = expand_to_rank(no, tf.rank(x_rg))
        y, h = self._channel([x_rg, no])

        # Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h)  # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no])  # LS channel estimation with nearest-neighbor
            
            h_perf = self._removed_null_subc(h)[0,0,0,0,0,0]
            # plt.figure()
            # plt.plot(np.real(h_perf))
            # plt.plot(np.imag(h_perf))
            # plt.plot(np.real(h_hat[0,0,0,0,0,0]), "--")
            # plt.plot(np.imag(h_hat[0,0,0,0,0,0]), "--")
            # plt.xlabel("Subcarrier index")
            # plt.ylabel("Channel frequency response")
            # plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
            # plt.title("Comparison of channel frequency responses")
            # plt.savefig('data/channel.png')
            
            x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no]) # LMMSE equalization
            no_eff_= expand_to_rank(no_eff, tf.rank(x_hat))
            llr = self._demapper([x_hat, no_eff_]) # Demapping
        elif self._system == "neural-receiver":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            llr = self._neural_receiver([y, no])
            llr = insert_dims(llr, 2, 1)  # Reshape the input to fit what the resource grid demapper is expected
            llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            llr = tf.reshape(llr, [batch_size, 1, 1, self._n])  # Reshape the LLRs to fit what the outer decoder is expected

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
