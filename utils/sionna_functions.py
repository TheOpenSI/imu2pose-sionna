import numpy as np
import sionna
# import sionna_vispy
import tensorflow as tf
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, CoverageMap
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies

def load_3d_map(map_name='floor_wall', render=True):
    # Load and return scene
    if map_name == 'munich':
        scene_name = sionna.rt.scene.munich
    elif map_name == 'etoile':
        scene_name = sionna.rt.scene.etoile
    else:
        scene_name = sionna.rt.scene.floor_wall
    scene = load_scene(scene_name)  # etoile, munich, floor_wall
    if render:
        render_scene(scene)
    return scene, scene_name

def render_scene(scene, paths=None, coverage_map=None):
    with sionna_vispy.patch():
        if paths is None:
            if coverage_map is None:
                scene.render(camera='scene-cam-0', num_samples=512)
                canvas = scene.preview()
            else:
                coverage_map.show()
                plt.show()
                # scene.render(camera='scene-cam-0', coverage_map=coverage_map, cm_metric='sinr',
                #              resolution=[480, 320])
                # canvas = scene.preview(coverage_map=coverage_map, cm_metric='sinr', cm_vmin=-10)
        else:
            # scene.render(camera='scene-cam-0', paths=paths, show_devices=True, show_paths=True, resolution=[480, 320])
            canvas = scene.preview(paths=paths, show_devices=True, show_paths=True)
    canvas.show()
    canvas.app.run()

def configure_antennas(scene, scene_name, num_tx_ant=8, num_rx_ant=4, position_tx=None, position_rx=None):
    # Configure antenna array for all transmitters
    # Note that the role of tx (user) and rx (base station) can be swapped
    scene.tx_array = PlanarArray(num_rows=1, num_cols=int(num_rx_ant/2), vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='tr38901', polarization='cross'
                                 )
    # configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1, num_cols=num_tx_ant, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='iso', polarization='V'
                                 )

    print('Number of TX antennas: {}'.format(scene.tx_array.num_ant))
    print('Number of RX antennas: {}'.format(scene.rx_array.num_ant))

    if position_tx is None:
        if scene_name.find('floor_wall') != -1:
            position_tx = [-2.5, 0.0, 6.0]
            position_rx = [-0.5, 0.0, 0.5]
        elif scene_name.find('etoile') != -1:
            position_tx = [-160.0, 70.0, 15.0]
            position_rx = [80.0, 70.0, 1.5]
        elif scene_name.find('munich') != -1:
            position_tx = [8.5, 21, 27] 
            position_rx = [45, 90, 1.5]
    # create transmitter
    tx = Transmitter(name='tx', position=position_tx)
    scene.add(tx)

    # create receiver
    rx = Receiver(name='rx', position=position_rx) # position will be updated later
    scene.add(rx)

    return scene

def configure_radio_material(scene, log=False):
    # Configure electromagnetic properties of objects based on
    # https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html#Ray-Tracing-for-Radio-Propagation
    scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True

    # Select an example object from the scene
    so = scene.get("Splendid__toile-itu_marble")  # Splendid__toile-itu_marble (etoile)  Altes_Rathaus-itu_marble
    # Print name of assigned radio material for different frequenies
    for f in [3.5e9, 2.14e9]:  # Print for differrent frequencies
        scene.frequency = f
        if log:
            print(f"\nRadioMaterial: {so.radio_material.name} @ {scene.frequency / 1e9:.2f}GHz")
            print("Conductivity:", so.radio_material.conductivity.numpy())
            print("Relative permittivity:", so.radio_material.relative_permittivity.numpy())
            print("Complex relative permittivity:", so.radio_material.complex_relative_permittivity.numpy())
            print("Relative permeability:", so.radio_material.relative_permeability.numpy())
            print("Scattering coefficient:", so.radio_material.scattering_coefficient.numpy())
            print("XPD coefficient:", so.radio_material.xpd_coefficient.numpy())
    return scene

def compute_paths_ray_tracing(scene):
    # Compute propagation paths with ray tracing
    paths = scene.compute_paths(max_depth=5, num_samples=1e6)
    return paths

def paths_to_cir(paths, reverse_direction=False, subcarrier_spacing=15e3, normalize_delay=False):
    # Convert from paths to channel impulse response
    # The last dimension corresponds to the number of time steps which defaults to one as there is no mobility
    print("Shape of `a` before applying Doppler shifts: ", paths.a.shape)

    paths.reverse_direction = reverse_direction
    # Apply Doppler shifts
    paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=14,
                        tx_velocities=[1, 0, 0], rx_velocities=[0, 0, 0]
                        )
    print("Shape of `a` after applying Doppler shifts: ", paths.a.shape)

    a, tau = paths.cir()
    print("Shape of tau: ", tau.shape)

    if not normalize_delay:
        # Disable normalization of delays
        paths.normalize_delays = False
        # Get only the LoS path
        a, tau = paths.cir(num_paths=75)
    else:
        paths.normalize_delays = True
        a, tau = paths.cir(num_paths=75)

    return a, tau

def cir_to_channels(a, tau, fft_size=48, subcarrier_spacing=15e3):
    # Convert CIR to channel states
    # Compute frequencies of subcarriers and center around carrier frequency
    frequencies = subcarrier_frequencies(num_subcarriers=fft_size, subcarrier_spacing=subcarrier_spacing)

    # Compute the frequency response of the channel at frequencies.
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)  # Non-normalized includes path-loss

    # Verify that the channel power is normalized
    h_avg_power = tf.reduce_mean(tf.abs(h_freq) ** 2).numpy()

    print("Shape of h_freq: ", h_freq.shape)
    print("Average power h_freq: ", h_avg_power)  # Channel is normalized

    return h_freq

def plot_cir(a, tau):
    t = tau[0, 0, 0, :] / 1e-9  # Scale to ns
    a_abs = np.abs(a)[0, 0, 0, 0, 0, :, 0]
    a_max = np.max(a_abs)
    # Add dummy entry at start/end for nicer figure
    t = np.concatenate([(0.,), t, (np.max(t) * 1.1,)])
    a_abs = np.concatenate([(np.nan,), a_abs, (np.nan,)])

    # Close all previous figures to avoid showing them
    plt.close('all')

    # And plot the CIR
    plt.figure()
    plt.title("Channel impulse response realization")

    plt.stem(t, a_abs)
    plt.xlim([0, np.max(t)])
    plt.ylim([-2e-6, a_max * 1.1])
    plt.xlabel(r"$\tau$ [ns]")
    plt.ylabel(r"$|a|$")
    plt.show()

def plot_h_freq(h_freq):
    # Extracting the magnitude of the first slice for visualization
    matrix_to_visualize = np.abs(h_freq[0, 0, 0, 0, 0, :, :])

    # Plotting the matrix
    plt.close('all')
    # Extracting the magnitude and phase of the first slice for visualization
    matrix_to_visualize = np.abs(h_freq[0, 0, 0, 0, 0, :, :])
    phase_matrix = np.angle(h_freq[0, 0, 0, 0, 0, :, :])

    # Plotting the magnitude and phase side by side
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Magnitude plot
    magnitude_plot = ax[0].imshow(matrix_to_visualize.T, aspect='auto', cmap='viridis')
    ax[0].set_title('Magnitude: Sub-Carriers vs. Time Steps')
    ax[0].set_xlabel('Time Steps')
    ax[0].set_ylabel('Sub-Carriers')
    fig.colorbar(magnitude_plot, ax=ax[0], label='Magnitude')

    # Phase plot
    phase_plot = ax[1].imshow(phase_matrix.T, aspect='auto', cmap='twilight')
    ax[1].set_title('Phase: Sub-Carriers vs. Time Steps')
    ax[1].set_xlabel('Time Steps')
    ax[1].set_ylabel('Sub-Carriers')
    fig.colorbar(phase_plot, ax=ax[1], label='Phase (radians)')

    plt.tight_layout()
    plt.show()

    # Correct extraction of the real and imaginary parts
    U = np.real(h_freq[0, 0, 0, 0, 0, :, :])
    V = np.imag(h_freq[0, 0, 0, 0, 0, :, :])

    # Swapping X and Y mesh grids to correct the orientation
    X, Y = np.meshgrid(np.arange(U.shape[0]), np.arange(U.shape[1]))

    # Plotting the corrected quiver plot
    plt.figure(figsize=(16, 8))
    plt.quiver(X, Y, U.T, V.T, scale=1, angles='xy', scale_units='xy',
               color='blue')  # Note the transpose to match orientation
    plt.title('Quiver Plot: Vector Field Representation of Complex Matrix')
    plt.xlabel('Time Steps')
    plt.ylabel('Sub-Carriers')
    plt.grid(True)
    plt.show()

def plot_time_series_waveform_for_one_time_step(h_freq, time_step=0):
    """
    Plot the time-domain waveform for a specific time step.

    Parameters:
    h_freq: numpy array
        Frequency response of the channel (from cir_to_channels()).
    time_step: int
        The time step to plot the waveform for.
    """
    # Extract the frequency response at a specific time step
    h_freq_at_time_step = h_freq[0, 0, 0, 0, 0, time_step, :]

    # Apply inverse FFT to convert to time-domain
    h_time_domain = np.fft.ifft(h_freq_at_time_step)

    # Create a time axis based on the number of sub-carriers
    time_axis = np.arange(h_time_domain.shape[0])

    # Plot the real part of the time-domain signal (which will resemble a sine wave)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, np.real(h_time_domain), label='Real Part')

    # Optionally plot the imaginary part
    plt.plot(time_axis, np.imag(h_time_domain), label='Imaginary Part', linestyle='--')

    plt.title(f"Time-Domain Waveform at Time Step {time_step}")
    plt.xlabel("Sample Index (Time)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_estimated_channel(h_perf, h_est):
    """
    Plot estimated channel state and perfect channel state with real and imaginary parts in separate subplots.
    :param h_perf: Channel with perfect CSI
    :param h_est: Estimated channel
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots side by side

    # Real part
    axs[0].plot(np.real(h_perf))
    axs[0].plot(np.real(h_est), '--')
    axs[0].set_xlabel("Subcarrier index")
    axs[0].set_ylabel("Channel frequency response (Real)")
    axs[0].legend(["Ideal (real part)", "Estimated (real part)"])
    axs[0].set_title("Real Part of Channel Frequency Response")

    # Imaginary part
    axs[1].plot(np.imag(h_perf))
    axs[1].plot(np.imag(h_est), '--')
    axs[1].set_xlabel("Subcarrier index")
    axs[1].set_ylabel("Channel frequency response (Imaginary)")
    axs[1].legend(["Ideal (imaginary part)", "Estimated (imaginary part)"])
    axs[1].set_title("Imaginary Part of Channel Frequency Response")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def inspect_paths(scene, paths, path_idx=2):
    # Get information from ray tracing paths
    print(" --- Information of the paths in the scene ----")

    # Show the coordinates of the starting points of all rays.
    print("Source coordinates: ", paths.sources.numpy())
    print("Transmitter coordinates: ", list(scene.transmitters.values())[0].position.numpy())

    # Show the coordinates of the endpoints of all rays.
    # These coincide with the location of the receivers.
    print("Target coordinates: ", paths.targets.numpy())
    print("Receiver coordinates: ", list(scene.receivers.values())[0].position.numpy())

    # Show the types of all paths:
    # 0 - LoS, 1 - Reflected, 2 - Diffracted, 3 - Scattered
    # Note that Diffraction and scattering are turned off by default.
    print("Path types: ", paths.types.numpy())

    # We can now access for every path the channel coefficient, the propagation delay,
    # as well as the angles of departure and arrival, respectively (zenith and azimuth).

    # Let us inspect a specific path in detail
    # path_idx = 2  # Try out other values in the range [0, 13]

    # For a detailed overview of the dimensions of all properties, have a look at the API documentation
    print(f"\n--- Detailed results for path {path_idx} ---")
    print(f"Channel coefficient: {paths.a[0, 0, 0, 0, 0, path_idx, 0].numpy()}")
    print(f"Propagation delay: {paths.tau[0, 0, 0, path_idx].numpy() * 1e6:.5f} us")
    print(f"Zenith angle of departure: {paths.theta_t[0, 0, 0, path_idx]:.4f} rad")
    print(f"Azimuth angle of departure: {paths.phi_t[0, 0, 0, path_idx]:.4f} rad")
    print(f"Zenith angle of arrival: {paths.theta_r[0, 0, 0, path_idx]:.4f} rad")
    print(f"Azimuth angle of arrival: {paths.phi_r[0, 0, 0, path_idx]:.4f} rad")


def test_sionna_functions():
    scene, scene_name = load_3d_map('etoile', render=False)  # floor_wall, etoile
    scene = configure_antennas(scene, scene_name)
    scene = configure_radio_material(scene)
    paths = compute_paths_ray_tracing(scene)
    a, tau = paths_to_cir(paths, normalize_delay=True)
    h_freq = cir_to_channels(a, tau)
    render_scene(scene, paths)
    plot_cir(a, tau)
    plot_h_freq(h_freq)
    plot_time_series_waveform_for_one_time_step(h_freq, 0)
    inspect_paths(scene, paths, path_idx=1)


if __name__ == '__main__':
    test_sionna_functions()


