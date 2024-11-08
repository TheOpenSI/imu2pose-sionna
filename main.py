import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from sionna.rt import PlanarArray, Receiver
from sionna.mimo import StreamManagement
from sionna.mimo.precoding import normalize_precoding_power, grid_of_beams_dft
from sionna.ofdm import (ResourceGrid, ResourceGridMapper, LSChannelEstimator, RemoveNulledSubcarriers,
                         LMMSEEqualizer, ZFPrecoder)
from sionna.mapping import Mapper, Demapper
from sionna.channel import CIRDataset, cir_to_ofdm_channel, subcarrier_frequencies, ApplyOFDMChannel, OFDMChannel
from sionna.utils import compute_ber, ebnodb2no, sim_ber, BinarySource
from utils.sionna_functions import (load_3d_map, configure_antennas, configure_radio_material,
                                    plot_h_freq, render_scene, plot_cir, plot_estimated_channel)
from utils.imu_functions import pre_processing_imu, imu_to_binary, binary_to_imu, numpy_to_tensorflow_source
from neural_receiver import E2ESystem

# tf.random.set_seed(1)
import matplotlib
matplotlib.use('QtAgg')

def evaluate_e2e_model(num_epochs=3000, gen_data=True, eval_mode=0):
    # End-to-end model
    ofdm_params = {
        # By default, Sionna uses TX to represents base station (with power_tx functions) 
        # and RX to represents UEs which can be randomly sampled with a coverage map.
        # For uplink transmission, just be careful with the order of TX-RX in ray tracing
        'num_rx_ant': 16,  # base station
        'num_tx_ant': 4,  # user's antenna
        'num_tx': 4,  # number of UEs
        'num_rx': 1, 
        'carrier_frequency': 3.5e9,  # Hz
        'delay_spread': 100e-9,  # s
        'speed': 10.0,  # Speed for evaluation and training [m/s]
        'ebno_db_min': -5.0,  # SNR range for evaluation and training [dB]
        'ebno_db_max': 10.0,
        'subcarrier_spacing': 15e3,  # Hz - OFDM waveform configuration
        'fft_size': 76,  # No of subcarriers in the resource grid, including the null-subcarrier and the guard bands
        'num_ofdm_symbols': 14,  # Number of OFDM symbols forming the resource grid
        'dc_null': True,  # Null the DC subcarrier
        'num_guard_carriers': [5, 6],  # Number of guard carriers on each side
        'pilot_pattern': "kronecker",  # Pilot pattern
        'pilot_ofdm_symbol_indices': [2, 11],  # Index of OFDM symbols carrying pilots
        'cyclic_prefix_length': 6,  # Simulation in frequency domain. This is useless
        'num_bits_per_symbol': 2,  # Modulation and coding configuration
    }
    
    model_params = {
        'quantization_level': 2**7,  # quanization level
        'batch_size': 128,  # batch size for training
        'num_batches': 10,  # number of training batches
        'imu_frame_per_batch_tx': 60,  # number of IMU data samples to be transmitted per batch_size transmission
    }

    # Create scene with transmitters/receivers
    scene, scene_name = load_3d_map(map_name='etoile', render=False)
    scene = configure_antennas(scene, scene_name, num_tx_ant=ofdm_params['num_tx_ant'], num_rx_ant=ofdm_params['num_rx_ant'])
    # if gen_data:
    #     sample_paths = scene.compute_paths(max_depth=5, num_samples=1e6)
    #     render_scene(scene, paths=sample_paths)
    #     del sample_paths

    model = E2ESystem('neural-receiver', ofdm_params, model_params, scene, eval_mode=eval_mode, gen_data=gen_data)
    optimizer = tf.keras.optimizers.legacy.Adam()
    ebno_db_min = ofdm_params['ebno_db_min']
    ebno_db_max = ofdm_params['ebno_db_max']

    if eval_mode == 0 or eval_mode == 1:
        if eval_mode == 1:
            # keep training the model from check point
            model(model_params['batch_size'], tf.constant(ebno_db_max, tf.float32))
            model_weights_path = 'data/neural_receiver_weights'
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            model.set_weights(weights)
        for i in range(1, num_epochs + 1):
            # Sampling a batch of SNRs
            ebno_db = tf.random.uniform(shape=[], minval=ebno_db_min, maxval=ebno_db_max)
            # Forward pass
            with tf.GradientTape() as tape:
                rate = model(model_params['batch_size'], ebno_db)
                loss = -rate
            # Computing and applying gradients
            weights = model.trainable_weights
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            # Periodically printing the progress
            if i % 10 == 0:
                print('Iteration {}/{}  Rate: {:.4f} bit'.format(i, num_epochs, rate.numpy()), end='\r')
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
            
            model = E2ESystem('baseline-perfect-csi', ofdm_params, model_params, scene, eval_mode=eval_mode, gen_data=False)
            _, bler = sim_ber(model, ebno_dbs, batch_size=model_params['batch_size'], num_target_block_errors=100, max_mc_iter=100, early_stop=True)
            BLER['baseline-perfect-csi'] = bler.numpy()
        
            model = E2ESystem('baseline-ls-estimation', ofdm_params, model_params, scene, eval_mode=eval_mode, gen_data=False)
            _, bler = sim_ber(model, ebno_dbs, batch_size=model_params['batch_size'], num_target_block_errors=100, max_mc_iter=100, early_stop=True)
            BLER['baseline-ls-estimation'] = bler.numpy()
            
            model = E2ESystem('neural-receiver', ofdm_params, model_params, scene, eval_mode=eval_mode, gen_data=False)
            model(model_params['batch_size'], tf.constant(ebno_db_max, tf.float32))
            model_weights_path = 'data/neural_receiver_weights'
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            model.set_weights(weights)
            _, bler = sim_ber(model, ebno_dbs, batch_size=model_params['batch_size'], num_target_block_errors=100, max_mc_iter=100, early_stop=True)
            BLER['neural-receiver'] = bler.numpy()

            plt.figure(figsize=(10, 6))
            # Baseline - Perfect CSI
            plt.semilogy(ebno_dbs, BLER['baseline-perfect-csi'], 'o-', c=f'C0', label=f'Baseline - Perfect CSI')
            # Baseline - LS Estimation
            plt.semilogy(ebno_dbs, BLER['baseline-ls-estimation'], 'x--', c=f'C1', label=f'Baseline - LS Estimation')
            # Neural receiver
            plt.semilogy(ebno_dbs, BLER['neural-receiver'], 's-.', c=f'C2', label=f'Neural receiver')
            plt.xlabel(r"$E_b/N_0$ (dB)")
            plt.ylabel("BLER")
            plt.grid(which="both")
            # plt.ylim((1e-4, 1.0))
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/ber.png')
        else:
            # We use customized data source with larget batch_size than 128
            batch_size = model.get_batch_size()
            model(batch_size, tf.constant(ebno_db_max, tf.float32))
            model_weights_path = 'data/neural_receiver_weights'
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            model.set_weights(weights)

            # Recover original IMU data
            original_batches = np.zeros(shape=(int(model_params['num_batches'] * model_params['imu_frame_per_batch_tx']), 204))  # (600, 204)
            recovered_batches = np.zeros(shape=(int(model_params['num_batches'] * model_params['imu_frame_per_batch_tx']), 204))  # (600, 204)
            print('Recovered IMU shape: {}'.format(recovered_batches.shape))

            for batch_id in range(model_params['num_batches']):
                # Recover original IMU data
                # One model forward is a transmission of `imu_frame_per_batch_tx`` IMU features
                data_source = model.get_binary_source()
                b, b_hat = model(batch_size, tf.constant(ebno_db_max, tf.float32), batch_id)  # b: [170, 1, 1, 1536] -> imu_shape: [10, 204]
                imu_shape = data_source.get_batch_imu_shape(batch_size=batch_size)  # [10, 204]
                b_flat = tf.reshape(b, [-1]).numpy().astype(np.int8)
                b_hat_flat = tf.reshape(b_hat, [-1]).numpy().astype(np.int8)
                print('Batch: {}/{}'.format(batch_id+1, model_params['num_batches']))
                print('--- original_imu_shape: {}'.format(imu_shape))
                print('--- b.shape: {}/{}'.format(b.shape, b.dtype))
                print('--- b_hat.shape: {}/{}'.format(b_hat.shape, b_hat.dtype))

                # Convert binary data to IMU data for this batch
                origin_data = binary_to_imu(b_flat, model_params['quantization_level'], imu_shape, -1.0, 1.0)  # [10, 204]
                recovered_data = binary_to_imu(b_hat_flat, model_params['quantization_level'], imu_shape, -1.0, 1.0)  # [10, 204]
                print('--- recovered_imu.shape: {}/{}'.format(recovered_data.shape, recovered_data.dtype))

                # Append the recovered data to the main arrays in the correct order
                start_idx = batch_id * imu_shape[0]
                end_idx = (batch_id + 1) * imu_shape[0]
                original_batches[start_idx:end_idx] = origin_data
                recovered_batches[start_idx:end_idx] = recovered_data
                np.save('data/ori_imu_{}.npy'.format(batch_id), origin_data)
                np.save('data/rec_imu_{}.npy'.format(batch_id), recovered_data)

            # Save the stacked array
            np.save('data/ori_imu_seq_{}.npy'.format(ebno_db_max), original_batches)
            np.save('data/rec_imu_seq_{}.npy'.format(ebno_db_max), recovered_batches)
            print('Recovered data shape: {}, {}'.format(recovered_batches.shape, recovered_batches.dtype))

if __name__ == '__main__':
    import argparse
    # Parse parameters
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--gen_data', type=int, help='Generate channel impulse response dataset', default=0)
    parser.add_argument('--num_ep', type=int, help='Number of training epochs', default=10000)
    parser.add_argument('--eval_mode', type=int, help='Training from scratch (0) - Training from check point (1) - BER evaluation (2) - Custom data forward (3)', default=0)
    args = parser.parse_args()

    evaluate_e2e_model(num_epochs=int(args.num_ep), gen_data=bool(args.gen_data), eval_mode=args.eval_mode)
