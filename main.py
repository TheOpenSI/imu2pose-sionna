import os
import numpy as np
import tensorflow as tf
from sionna.mimo import StreamManagement
from sionna.mimo.precoding import normalize_precoding_power, grid_of_beams_dft
from sionna.ofdm import (ResourceGrid, ResourceGridMapper, LSChannelEstimator, RemoveNulledSubcarriers,
                         LMMSEEqualizer, ZFPrecoder)
from sionna.mapping import Mapper, Demapper
from sionna.channel import CIRDataset, cir_to_ofdm_channel, subcarrier_frequencies, ApplyOFDMChannel
from sionna.utils import compute_ber, ebnodb2no
from utils.sionna_functions import (load_3d_map, configure_antennas, configure_radio_material,
                                    plot_h_freq, render_scene, plot_cir, plot_estimated_channel)
from utils.imu_functions import pre_processing_imu, imu_to_binary, binary_to_imu, numpy_to_tensorflow_source

tf.random.set_seed(1)

# Class from https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#BER-Evaluation
class CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses.

    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.

    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.

    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter.

    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter.

    num_tx : int
        Number of transmitters

    reverse_direction: bool
        If `reverse_direction` is `True`, the direction is uplink

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

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

def generate_channel_batch(scene, batch_size, num_paths, rg, sampling_frequency, frequencies, save=True):
    # Generating batch_size channel frequency responses
    a, tau = [], []
    print('Generating batches of h_freq ... ')  # will take some time
    if save:
        for i in range(batch_size):
            print('Progress: {}/{}'.format(i, batch_size), end='\r')
            paths_i = scene.compute_paths(max_depth=5, num_samples=1e6)
            paths_i.apply_doppler(sampling_frequency=sampling_frequency, num_time_steps=rg.num_ofdm_symbols)
            paths_i.normalize_delays = True
            a_i, tau_i = paths_i.cir(num_paths=num_paths)
            a_i = np.transpose(a_i, [3, 1, 2, 0, 4, 5, 6])
            tau_i = np.transpose(tau_i, [2, 1, 0, 3])
            # Remove CIRs that have no active link (i.e., a is all-zero)
            p_link = np.sum(np.abs(a_i) ** 2, axis=(1, 2, 3, 4, 5, 6))
            a_i = a_i[p_link > 0., ...]
            tau_i = tau_i[p_link > 0., ...]
            a.append(a_i[0])
            tau.append(tau_i[0])
        # shape of a # [batch_size, num_tx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_ofdm_symbols]
        a = np.stack(a)
        tau = np.stack(tau)
        print('a.shape: {}'.format(a.shape))
        print('tau.shape: {}'.format(tau.shape))
        h = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
        np.save('data/channel_dataset.npy', h.numpy())
    else:
        h = tf.convert_to_tensor(np.load('data/channel_dataset.npy'), dtype=tf.complex64)
    return h


def configure_links(scene, num_tx, num_rx):
    """
    Set up communication links between transmitter and receiver.
    :param scene: sionna.rt.Scene object
    :return: transmitter, receiver, channel
    """
    # OFDM resource grid configuration
    num_ut_ant = scene.rx_array.num_ant
    num_bs_ant = scene.tx_array.num_ant
    num_streams_per_tx = num_ut_ant

    # Stream management between tx and rx
    rx_tx_association = np.array([[1]])
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    # Resource grid configuration
    rg = ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=48,
        subcarrier_spacing=15e3,
        num_tx=num_tx,
        num_streams_per_tx=num_streams_per_tx,
        cyclic_prefix_length=6,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11]
    )

    # Precoding
    zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)
    remove_nulled_scs = RemoveNulledSubcarriers(rg)

    # Channel coding configuration
    num_bits_per_symbol = 2

    # OFDM modulator
    mapper = Mapper('qam', num_bits_per_symbol)
    rg_mapper = ResourceGridMapper(rg)

    # Channel estimator
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # Demodulator
    demapper = Demapper("app", "qam", num_bits_per_symbol)

    return mapper, rg, rg_mapper, zf_precoder, ls_est, lmmse_equ, demapper, remove_nulled_scs

def prepare_source_data(shape, quantization_level):
    # load IMU data
    path = os.path.expanduser('~/datasets/DIP_IMU_and_Others/processed_test.npz')
    data = np.load(path)['imu']
    data = np.squeeze(data)
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

def uplink_transmission():
    # Configure scene and antennas
    num_tx = 1
    num_rx = 1
    num_tx_ant = 8
    num_rx_ant = 1
    scene, scene_name = load_3d_map(map_name='etoile', render=False)
    scene = configure_antennas(scene, scene_name, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant)
    scene = configure_radio_material(scene)
    sample_paths = scene.compute_paths(max_depth=5, num_samples=1e6)
    render_scene(scene, paths=sample_paths)

    # Initialize configurations for source coding
    imu_seq_len = 10  # number of IMU data samples to be transmitted
    batch_size = 204 * imu_seq_len
    quantization_level = 2 ** 7
    uplink_direction = False  # Reverse direction for uplink transmission
    perfect_csi = False
    precoding = True

    # Initialize modulation and channel estimation
    (mapper, rg, rg_mapper, zf_precoder, ls_est,
     lmmse_equ, demapper, remove_nulled_scs) = configure_links(scene, num_tx, num_rx)

    # Customized channel
    sampling_frequency = 15e3
    num_paths = 75
    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    h_freq = generate_channel_batch(scene, batch_size, num_paths, rg, sampling_frequency, frequencies, save=False)
    print('h_freq.shape: {}'.format(h_freq.shape))
    channel_freq = ApplyOFDMChannel(add_awgn=True)

    # Start transmission
    num_bits_per_symbols = 2
    ebno_db = 1.0
    no = ebnodb2no(ebno_db, num_bits_per_symbols, 1.0, rg)

    # We want to transmit 204 features of on IMU data sample per slot, each IMU sample is quantized
    # with quantization_level bits.
    # Define number of information bits per OFDM resource grid with 2 pilot slots, fft_size=48, num_tx=1
    k = int(rg.num_data_symbols * num_bits_per_symbols)
    data_shape = (batch_size, 1, rg.num_streams_per_tx, k)
    print('Desired TX data shape: {}'.format(data_shape))
    b, imu = prepare_source_data(shape=data_shape, quantization_level=quantization_level)
    print('b.shape: {}'.format(b.shape))
    x = mapper(b)
    print('x.shape: {}'.format(x.shape))
    x_rg = rg_mapper(x)
    print('x_rg.shape: {}'.format(x_rg.shape))

    # Precoding for downlink transmission
    g = None
    if not uplink_direction:
        x_rg, g = zf_precoder([x_rg, h_freq])
    y = channel_freq([x_rg, h_freq, no])  # (b, 1, 1, 14, 48)
    print('y.shape: {}'.format(y.shape))

    # Decode and channel estimation
    if perfect_csi:
        if not uplink_direction:
            if precoding:
                h_hat = g
            else:
                h_hat = h_freq
        else:
            h_hat = h_freq
        err_var = 0.0
    else:
        h_hat, err_var = ls_est([y, no])
    print('h_err mean: {}'.format(np.square(h_hat - h_freq).mean()))
    if not precoding:
        plot_estimated_channel(h_freq[0, 0, 0, 0, 0, 0], h_hat[0, 0, 0, 0, 0, 0])

    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])  # h_hat = h_freq???
    print("Shape of x_hat: ", x_hat.shape)
    print("Shape of no_eff: ", no_eff.shape)
    llr = demapper([x_hat, no_eff])
    print("Shape of llr: {}, {} ".format(llr.shape, llr.dtype))
    b_hat = tf.where(llr > 0.0, 1.0, 0.0)
    print("Shape of b_hat: {}, {} ".format(b_hat.shape, b_hat.dtype))
    print('BER: {} - Precoding: {} - Perfect_CSI: {}'.format(compute_ber(b, b_hat), precoding, perfect_csi))

    # Recover original IMU data
    b_hat_flat = tf.cast(tf.reshape(b_hat, [-1]), dtype=tf.int8).numpy()
    print('b_hat_flat.shape: {}'.format(b_hat_flat.shape))
    recovered_data = binary_to_imu(b_hat_flat, quantization_level, imu.shape, -1.0, 1.0)
    print('recovered_data.shape: {}, {}'.format(recovered_data.shape, recovered_data.dtype))
    rec_flat_len = np.prod(recovered_data.shape)
    recovered_imu_seq = np.reshape(recovered_data, (int(rec_flat_len/204), 204))
    # np.save('data/recovered_imu_seq_{}.npy'.format(ebno_db), recovered_imu_seq)


if __name__ == '__main__':
    uplink_transmission()
