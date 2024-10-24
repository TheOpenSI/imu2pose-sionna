import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber

sionna.config.seed = 42


def simulation():
    num_ut = 1
    num_bs = 1
    num_ut_ant = 4
    num_bs_ant = 8

    num_streams_per_tx = num_ut_ant
    rx_tx_association = np.array([[1]])
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    rg = ResourceGrid(num_ofdm_symbols=14,
                      fft_size=76,
                      subcarrier_spacing=15e3,
                      num_tx=1,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=6,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])
    rg.show()
    plt.show()

    carrier_frequency = 2.6e9  # Carrier frequency in Hz.
    # This is needed here to define the antenna element spacing.

    ut_array = AntennaArray(num_rows=1,
                            num_cols=int(num_ut_ant / 2),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=int(num_bs_ant / 2),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    delay_spread = 300e-9  # Nominal delay spread in [s]. Please see the CDL documentation
    # about how to choose this value.

    direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
    # In the `uplink`, the UT is transmitting.
    cdl_model = "B"  # Suitable values are ["A", "B", "C", "D", "E"]

    speed = 1  # UT speed [m/s]. BSs are always assumed to be fixed.
    # The direction of travel will chosen randomly within the x-y plane.

    # Configure a channel impulse reponse (CIR) generator for the CDL model.
    # cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
    cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)

    a, tau = cdl(batch_size=32, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1 / rg.ofdm_symbol_duration)

    print("Shape of the path gains: ", a.shape)
    print("Shape of the delays:", tau.shape)

    plt.figure()
    plt.title("Channel impulse response realization")
    plt.stem(tau[0, 0, 0, :] / 1e-9, np.abs(a)[0, 0, 0, 0, 0, :, 0])
    plt.xlabel(r"$\tau$ [ns]")
    plt.ylabel(r"$|a|$")

    plt.figure()
    plt.title("Time evolution of path gain")
    plt.plot(np.arange(rg.num_ofdm_symbols) * rg.ofdm_symbol_duration / 1e-6, np.real(a)[0, 0, 0, 0, 0, 0, :])
    plt.plot(np.arange(rg.num_ofdm_symbols) * rg.ofdm_symbol_duration / 1e-6, np.imag(a)[0, 0, 0, 0, 0, 0, :])
    plt.legend(["Real part", "Imaginary part"])

    plt.xlabel(r"$t$ [us]")
    plt.ylabel(r"$a$")
    plt.show()

    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

    plt.figure()
    plt.title("Channel frequency response")
    plt.plot(np.real(h_freq[0, 0, 0, 0, 0, 0, :]))
    plt.plot(np.imag(h_freq[0, 0, 0, 0, 0, 0, :]))
    plt.xlabel("OFDM Symbol Index")
    plt.ylabel(r"$h$")
    plt.legend(["Real part", "Imaginary part"])
    plt.show()

    # Function that will apply the channel frequency response to an input signal
    channel_freq = ApplyOFDMChannel(add_awgn=True)

    # The following values for truncation are recommended.
    # Please feel free to tailor them to you needs.
    l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
    l_tot = l_max - l_min + 1

    h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)
    # Function that will apply the discrete-time channel impulse response to an input signal
    channel_time = ApplyTimeChannel(rg.num_time_samples, l_tot=l_tot, add_awgn=True)

    a, tau = cdl(batch_size=2, num_time_steps=rg.num_time_samples + l_tot - 1, sampling_frequency=rg.bandwidth)

    num_bits_per_symbol = 2  # QPSK modulation
    coderate = 0.5  # Code rate
    n = int(rg.num_data_symbols * num_bits_per_symbol)  # Number of coded bits
    k = int(n * coderate)  # Number of information bits

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(k, n)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", num_bits_per_symbol)

    # The resource grid mapper maps symbols onto an OFDM resource grid
    rg_mapper = ResourceGridMapper(rg)

    # The zero forcing precoder precodes the transmit stream towards the intended antennas
    zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)

    # OFDM modulator and demodulator
    modulator = OFDMModulator(rg.cyclic_prefix_length)
    demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)

    # This function removes nulled subcarriers from any tensor having the shape of a resource grid
    remove_nulled_scs = RemoveNulledSubcarriers(rg)

    # The LS channel estimator will provide channel estimates and error variances
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("app", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    batch_size = 400  # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
    ebno_db = 30
    perfect_csi = True

    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
    c = encoder(b)
    x = mapper(c)
    x_rg = rg_mapper(x)

    # The CIR needs to be sampled every 1/bandwith [s].
    # In contrast to frequency-domain modeling, this implies
    # that the channel can change over the duration of a single
    # OFDM symbol. We now also need to simulate more
    # time steps.
    cir = cdl(batch_size, rg.num_time_samples + l_tot - 1, rg.bandwidth)

    # OFDM modulation with cyclic prefix insertion
    x_time = modulator(x_rg)

    # Compute the discrete-time channel impulse reponse
    h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)

    # Compute the channel output
    # This computes the full convolution between the time-varying
    # discrete-time channel impulse reponse and the discrete-time
    # transmit signal. With this technique, the effects of an
    # insufficiently long cyclic prefix will become visible. This
    # is in contrast to frequency-domain modeling which imposes
    # no inter-symbol interfernce.
    y_time = channel_time([x_time, h_time, no])

    # OFDM demodulation and cyclic prefix removal
    y = demodulator(y_time)

    if perfect_csi:

        a, tau = cir

        # We need to sub-sample the channel impulse reponse to compute perfect CSI
        # for the receiver as it only needs one channel realization per OFDM symbol
        a_freq = a[..., rg.cyclic_prefix_length:-1:(rg.fft_size + rg.cyclic_prefix_length)]
        a_freq = a_freq[..., :rg.num_ofdm_symbols]

        # Compute the channel frequency response
        h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)

        h_hat, err_var = remove_nulled_scs(h_freq), 0.
    else:
        h_hat, err_var = ls_est([y, no])

    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
    llr = demapper([x_hat, no_eff])
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat)
    print("BER: {}".format(ber))

    # In the example above, we assumed perfect CSI, i.e.,
    # h_hat correpsond to the exact ideal channel frequency response.
    h_perf = h_hat[0, 0, 0, 0, 0, 0]

    # We now compute the LS channel estimate from the pilots.
    h_est, _ = ls_est([y, no])
    h_est = h_est[0, 0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(np.real(h_perf))
    plt.plot(np.imag(h_perf))
    plt.plot(np.real(h_est), "--")
    plt.plot(np.imag(h_est), "--")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Channel frequency response")
    plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"])
    plt.title("Comparison of channel frequency responses")
    plt.show()


if __name__ == '__main__':
    simulation()
