import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy
from sionna.utils import sim_ber, compute_ber
from sionna.ofdm import ResourceGrid
from sionna.rt import PlanarArray, Receiver, Transmitter, Camera
from utils.sionna_functions import (load_3d_map, render_scene, configure_antennas)
from utils.imu_functions import binary_to_imu
from neural_receiver import E2ESystem

# Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for i in range(len(gpus)): 
        try:
            tf.config.experimental.set_memory_growth(gpus[i], True)
        except RuntimeError as e:
            print(e)
tf.get_logger().setLevel('ERROR')

print("Num GPUs Available: ", len(gpus))

def generate_channel_impulse_responses(scene, map_name, num_cirs, batch_size_cir, rg, num_tx_ant, num_rx_ant, num_paths, uplink=True):
    max_depth = 5
    min_gain_db = -130  # in dB / ignore any position with less than -130 dB path gain
    max_gain_db = 0  # in dB / ignore any position with more than 0 dB path gain
    # Sample points within a 10-400m radius around the transmitter
    min_dist = 10  # in m
    max_dist = 400  # in m
    
    if map_name == 'etoile':
        tx_position = [-160.0, 70.0, 15.0]
        rx_position = [80.0, 70.0, 1.5]
        tx_look_at = [0, 0, 0]
        my_cam = Camera("my_cam", position=[-350, 250, 350], look_at=[-20, 0, 0])
    elif map_name == 'munich':
        tx_position = [-210, 73, 105] # [-210, 73, 105] / [8.5, 21, 27]]
        rx_position = [55, 80, 1.5]
        tx_look_at = rx_position
        my_cam = Camera("my_cam", position=[-400, 250, 150], look_at=[-15, 30, 28])
        
    scene = configure_antennas(
        scene, map_name, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, 
        position_tx=tx_position, position_rx=rx_position
        )
    scene.add(my_cam)
    sample_paths = scene.compute_paths(max_depth=5, num_samples=1e6)
    scene.render_to_file(
        "my_cam", paths=sample_paths, show_devices=True, show_paths=True, 
        resolution=[650, 500], filename='data/scene_{}.png'.format(map_name)
        )
    print('Rendered ray tracing scene to data/scene_{}.png'.format(map_name))
    # render_scene(scene, paths=sample_paths)
    del sample_paths
    
    # Remove old tx from scene
    scene.remove('tx')
    scene.synthetic_array = True # Emulate multiple antennas to reduce ray tracing complexity
    scene.tx_array = PlanarArray(num_rows=1,
                            num_cols=int(num_rx_ant/2), # We want to transmitter to be equiped with the 16 rx antennas
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="tr38901",
                            polarization="cross")
    # Create transmitter
    tx = Transmitter(name="tx",
                    position=tx_position,  
                    look_at=tx_look_at) # optional, defines view direction
    scene.add(tx)
    
    # Update coverage map
    print('Generating channel dataset and 3D map ...')
    print('Update coverage map ...')
    cm = scene.coverage_map(max_depth=max_depth, diffraction=True, cm_cell_size=(1.0, 1.0), combining_vec=None,
                            precoding_vec=None, num_samples=int(1e6)
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
    
    # remove current RX (user) and then simulating multiple random-positinoed RXs (users) later with ray tracing
    scene.remove("rx")  
    for i in range(batch_size_cir):
        scene.remove(f"rx-{i}")
        
    scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=num_tx_ant, # We want to transmitter to be equiped with the 16 rx antennas
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")  # Single antenna
        
    for i in range(batch_size_cir):
        rx = Receiver(name=f"rx-{i}",
                        position=ue_pos[i],  # Random position sampled from coverage map
                        )
        scene.add(rx)
        
    # scene.render_to_file("birds_view", show_devices=True, resolution=[650, 500], filename='data/user_positions.png')

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
        paths.normalize_delays = True
        paths.reverse_direction = uplink  # Convert to uplink direction
        paths.apply_doppler(sampling_frequency=rg.subcarrier_spacing,
                            num_time_steps=rg.num_ofdm_symbols,
                            tx_velocities=[0.0, 0.0, 0.0],
                            rx_velocities=[4.0, 3.0, 0.0])

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
    a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])  # [3, 1, 2, 0, 4, 5, 6]
    tau = np.transpose(tau, [2, 1, 0, 3])  # [2, 1, 0, 3]

    # Remove CIRs that have no active link (i.e., a is all-zero)
    p_link = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5, 6))
    a = a[p_link > 0., ...]
    tau = tau[p_link > 0., ...]
    
    # Remove CIRs that have invalid shapes
    a_list = [a[i] for i in range(a.shape[0])]
    tau_list = [tau[i] for i in range(tau.shape[0])]

    # Filter out elements with invalid shape
    valid_a_list = [arr for arr in a_list if len(arr.shape) == 6]
    valid_tau_list = [arr for arr in tau_list if len(arr.shape) == 3]

    # Convert back to a numpy array if needed
    valid_a_array = np.array(valid_a_list)
    valid_tau_array = np.array(valid_tau_list)

    np.save('data/a_dataset_{}.npy'.format(map_name), valid_a_array)
    np.save('data/tau_dataset_{}.npy'.format(map_name), valid_tau_array)

    return a, tau

def mse_simulation(quantization_range, ofdm_params, model_params, a, tau):
    MSE = {}
    ofdm_params['ebno_db_max'] = 10.0
    
    for system in ['neural-receiver', 'baseline-ls-estimation', 'baseline-perfect-csi']: 
        print('MSE evaluation on {}'.format(system))   
        mse_system, ber_system = [], []
        for i, ql in enumerate(quantization_range):
            print('-- Quantization level: {}'.format(ql))
            model_params['quantization_level'] = 2**ql
            
            model = E2ESystem(system, ofdm_params, model_params, a, tau, eval_mode=3, gen_data=False)
            batch_size = model.get_batch_size()
            if system == 'neural-receiver':
                model(batch_size, tf.constant(ofdm_params['ebno_db_max'], tf.float32))
                model_weights_path = 'data/neural_receiver_weights'
                with open(model_weights_path, 'rb') as f:
                    weights = pickle.load(f)
                model.set_weights(weights)
            
            binary_source = model.get_binary_source()
            b_all, b_hat_all = [], []
            for batch_id in range(binary_source.num_ofdm_rg_batches):
                
                b, b_hat = model(batch_size, tf.constant(ofdm_params['ebno_db_max'], tf.float32), batch_id)
                b_all.append(b)
                b_hat_all.append(b_hat)
            b_all = np.concatenate(np.asarray(b_all, dtype=int), axis=0)
            b_hat_all = np.concatenate(np.asarray(b_hat_all, dtype=int), axis=0)
            origin_data = binary_source.source_imu_original
            quantized_data = binary_source.source_imu_quantized
            data_min = np.min(origin_data, axis=0)
            data_max = np.max(origin_data, axis=0)
            recovered_data = binary_to_imu(b_hat_all, model_params['quantization_level'], quantized_data.shape, data_min, data_max)
            print('b_all.shape: {}'.format(b_all.shape))
            print('b_hat_all.shape: {}'.format(b_hat_all.shape))
            
            # Print results
            mse_i = np.mean((quantized_data - recovered_data)**2)
            ber_i = compute_ber(b_all, b_hat_all).numpy()
            mse_system.append(mse_i)
            ber_system.append(ber_i)
            # print some samples to see if recovered data is correct
            print('quantized data: {}'.format(quantized_data[:2, :10]))
            print('recovered data: {}'.format(recovered_data[:2, :10]))
            
            # save results
            np.save('data/ori_imu_{}_{}_{}.npy'.format(system, ql, ofdm_params['ebno_db_max']), origin_data)
            np.save('data/qtz_imu_{}_{}_{}.npy'.format(system, ql, ofdm_params['ebno_db_max']), quantized_data)
            np.save('data/rec_imu_{}_{}_{}.npy'.format(system, ql, ofdm_params['ebno_db_max']), recovered_data)
            
        print('---- MSE {}: {}'.format(system, np.mean(mse_system)))    
        print('---- BER: {}: {}'.format(system, np.mean(ber_system)))
        
        MSE[system] = mse_system
        
    plt.figure(figsize=(10, 6))
    # Neural receiver
    plt.semilogy(quantization_range, MSE['neural-receiver'], 's-', c=f'C0', label=f'Neural Receiver')
    # Baseline - LS Estimation
    plt.semilogy(quantization_range, MSE['baseline-ls-estimation'], '*--', c=f'C1', label=f'Baseline - LS Estimation')
    # Baseline - Perfect CSI
    plt.semilogy(quantization_range, MSE['baseline-perfect-csi'], 'o--', c=f'C2', label=f'Baseline - Perfect CSI')
    plt.xlabel("Quatization level")
    plt.ylabel("MSE")
    plt.grid(which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/mse.png')
        
    print(MSE)
    

def evaluate_e2e_model(num_epochs=3000, gen_data=True, eval_mode=0):
    # End-to-end model
    ofdm_params = {
        'num_rx_ant': 16,  # base station
        'num_tx_ant': 1,  # user's antenna - single Vertical antenna
        'num_tx': 1,  # number of UEs
        'num_rx': 1, 
        'ebno_db_min': -5.0,  # SNR range for evaluation and training [dB]
        'ebno_db_max': 16.0,
        'subcarrier_spacing': 30e3,  # Hz - OFDM waveform configuration
        'fft_size': 128,  # No of subcarriers in the resource grid, including the null-subcarrier and the guard bands
        'num_ofdm_symbols': 14,  # Number of OFDM symbols forming the resource grid
        'dc_null': True,  # Null the DC subcarrier
        'num_guard_carriers': [5, 6],  # Number of guard carriers on each side
        'pilot_pattern': "kronecker",  # Pilot pattern
        'pilot_ofdm_symbol_indices': [2, 11],  # Index of OFDM symbols carrying pilots
        'cyclic_prefix_length': 0,  # Simulation in frequency domain. This is useless
        'num_bits_per_symbol': 2,  # Modulation and coding configuration
        'num_rt_paths': 75, # Number of ray tracing paths in simulation
    }
    
    model_params = {
        'quantization_level': 2**8,  # quanization level
        'batch_size': 100,  # batch size for OFDM transmission
        'num_imu_frames': 6000,  # number of IMU frames 
    }

    if gen_data:
        # Create scene with transmitters/receivers
        # Creating all scenes and datasets
        for map_name in ['etoile', 'munich']:
            scene, _ = load_3d_map(map_name=map_name, render=False)
            # Customized channel
            rg = ResourceGrid(
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

            a, tau = generate_channel_impulse_responses(
                scene, map_name, 6000, 100, rg, 
                ofdm_params['num_tx_ant'], ofdm_params['num_rx_ant'], 
                ofdm_params['num_rt_paths'], True
                )
            print('a_{}.shape: {}'.format(map_name, a.shape))
            print('tau_{}.shape: {}'.format(map_name, tau.shape))
        sys.exit()
    else:
        a_dataset_etoile = np.load('data/a_dataset_etoile.npy')
        tau_dataset_etoile = np.load('data/tau_dataset_etoile.npy')
        a_dataset_munich = np.load('data/a_dataset_munich.npy')
        tau_dataset_munich = np.load('data/tau_dataset_munich.npy')
        
        # Combine the datasets along the first dimension
        a = np.concatenate((a_dataset_etoile, a_dataset_munich), axis=0)
        tau = np.concatenate((tau_dataset_etoile, tau_dataset_munich), axis=0)
        
        # Shuffle the datasets together
        indices = np.arange(a.shape[0])
        np.random.shuffle(indices)
        a = a[indices]
        tau = tau[indices]

        # Display the shapes of the combined datasets
        print("Shape of a_dataset:", a.shape)
        print("Shape of tau_dataset:", tau.shape)
              
    model = E2ESystem('neural-receiver', ofdm_params, model_params, a, tau, eval_mode=eval_mode, gen_data=gen_data)
    optimizer = tf.keras.optimizers.legacy.Adam()
    ebno_db_min = ofdm_params['ebno_db_min']
    ebno_db_max = ofdm_params['ebno_db_max']

    if eval_mode == 0 or eval_mode == 1:
        if eval_mode == 1:
            # keep training the model from check point
            ebno_db = tf.random.uniform(shape=[model_params['batch_size']], minval=ebno_db_min, maxval=ebno_db_max)
            model(model_params['batch_size'], ebno_db)
            model_weights_path = 'data/neural_receiver_weights'
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            model.set_weights(weights)
        rate_values = []
        for i in range(1, num_epochs + 1):
            # Sampling a batch of SNRs
            ebno_db = tf.random.uniform(shape=[model_params['batch_size']], minval=ebno_db_min, maxval=ebno_db_max)
            with tf.GradientTape() as tape:
                rate = model(model_params['batch_size'], ebno_db)
                loss = -rate
            # Computing and applying gradients
            weights = model.trainable_weights
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            # Periodically printing the progress
            if i % 10 == 0:
                print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_epochs, rate.numpy()))
                rate_values.append(rate.numpy())
                plt.plot(np.arange(len(rate_values)), rate_values, '-b')
                plt.xlabel('Iteration (x10)')
                plt.ylabel('Rate')
                plt.savefig('data/rate_plot.png')
                
                weights = model.get_weights()
                model_weights_path = 'data/neural_receiver_weights'
                with open(model_weights_path, 'wb') as f:
                    pickle.dump(weights, f)
    else:
        if eval_mode == 2:
            # BER simulation 
            ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                                 ebno_db_max, # Max SNR for evaluation
                                 1.0) # Step
            
            BLER = {}
            
            # Neural receiver
            model = E2ESystem('neural-receiver', ofdm_params, model_params, a, tau, eval_mode=eval_mode, gen_data=False)
            model(model_params['batch_size'], tf.constant(ebno_db_max, tf.float32))
            model_weights_path = 'data/neural_receiver_weights'
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            model.set_weights(weights)
            ber, bler = sim_ber(model, ebno_dbs, batch_size=model_params['batch_size'], num_target_block_errors=100, max_mc_iter=100, early_stop=True)
            BLER['neural-receiver'] = bler.numpy()
            
            # LS estimation
            model = E2ESystem('baseline-ls-estimation', ofdm_params, model_params, a, tau, eval_mode=eval_mode, gen_data=False)
            ber, bler = sim_ber(model, ebno_dbs, batch_size=model_params['batch_size'], num_target_block_errors=100, max_mc_iter=100, early_stop=True)
            BLER['baseline-ls-estimation'] = bler.numpy()
            
            # perfect CSI
            model = E2ESystem('baseline-perfect-csi', ofdm_params, model_params, a, tau, eval_mode=eval_mode, gen_data=False)
            ber, bler = sim_ber(model, ebno_dbs, batch_size=model_params['batch_size'], num_target_block_errors=100, max_mc_iter=100, early_stop=True)
            BLER['baseline-perfect-csi'] = bler.numpy()

            plt.figure(figsize=(10, 6))
            # Neural receiver
            plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-', c=f'C0', label=f'Neural Receiver')
            # Baseline - LS Estimation
            plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], '*--', c=f'C1', label=f'Baseline - LS Estimation')
            # Baseline - Perfect CSI
            plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o--', c=f'C2', label=f'Baseline - Perfect CSI')
            plt.xlabel(r"$E_b/N_0$ (dB)")
            plt.ylabel("BLER")
            plt.grid(which="both")
            plt.ylim((1e-4, 1.0))
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/ber.png')
        else:
            # MSE simulation with customized IMU data
            quantz_range = np.arange(6, 12, 2, dtype=int)
            mse_simulation(quantz_range, ofdm_params, model_params, a, tau)

if __name__ == '__main__':
    import argparse
    # Parse parameters
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--gen_data', type=int, help='Generate channel impulse response dataset', default=0)
    parser.add_argument('--num_ep', type=int, help='Number of training epochs', default=150000)
    parser.add_argument('--eval_mode', type=int, help='Training from scratch (0) - Training from check point (1) - BER evaluation (2) - Custom data forward (3)', default=0)
    args = parser.parse_args()

    evaluate_e2e_model(num_epochs=int(args.num_ep), gen_data=bool(args.gen_data), eval_mode=args.eval_mode)
