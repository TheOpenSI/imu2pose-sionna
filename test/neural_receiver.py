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
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

def main():
    filename='/home/hinguyen/Data/PycharmProjects/imu2pose-sionna/data/scene.png'
    # Scene
    scene = load_scene(sionna.rt.scene.munich)
    my_cam = Camera("my_cam", position=[-250,250,150], look_at=[-15,30,28])
    scene.add(my_cam)
    scene.render_to_file(camera="scene-cam-0", resolution=[650, 500], filename=filename)
    
    # Antennas
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")

    # Create transmitter
    tx = Transmitter(name="tx",
                    position=[8.5,21,27])

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a receiver
    rx = Receiver(name="rx",
                position=[45,90,1.5],
                orientation=[0,0,0])

    # Add receiver instance to scene
    scene.add(rx)

    tx.look_at(rx) # Transmitter points towards receiver
    
    # Compute propagation paths
    paths = scene.compute_paths(max_depth=5,
                                num_samples=1e6)  # Number of rays shot into directions defined
                                                # by a Fibonacci sphere , too few rays can
                                                # lead to missing paths
    paths.normalize_delays = True

    # Visualize paths in the scene
    scene.render_to_file("my_cam", paths=paths, show_devices=True, show_paths=True, resolution=[650, 500], filename=filename)
    
    # Default parameters in the PUSCHConfig
    subcarrier_spacing = 30e3
    fft_size = 48
    num_time_steps = 14
    num_tx = 4 # Number of users
    num_rx = 1 # Only one receiver considered
    num_tx_ant = 4 # Each user has 4 antennas
    num_rx_ant = 16 # The receiver is equipped with 16 antennas

    # batch_size for CIR generation
    batch_size_cir = 1000
    
    # Remove old tx from scene
    scene.remove("tx")

    scene.synthetic_array = True # Emulate multiple antennas to reduce ray tracing complexity
    # Transmitter (=basestation) has an antenna pattern from 3GPP 38.901
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=int(num_rx_ant/2), # We want to transmitter to be equiped with the 16 rx antennas
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="cross")

    # Create transmitter
    tx = Transmitter(name="tx",
                    position=[8.5,21,27],
                    look_at=[45,90,1.5]) # optional, defines view direction
    scene.add(tx)
    
    max_depth = 5 # Defines max number of ray interactions

    # Update coverage_map
    cm = scene.coverage_map(max_depth=max_depth,
                            diffraction=True,
                            cm_cell_size=(1., 1.),
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(1e6))
    
    min_gain_db = -130 # in dB; ignore any position with less than -130 dB path gain
    max_gain_db = 0 # in dB; ignore strong paths

    # sample points in a 5-400m radius around the receiver
    min_dist = 5 # in m
    max_dist = 400 # in m

    #sample batch_size random user positions from coverage map
    ue_pos, _ = cm.sample_positions(num_pos=batch_size_cir,
                                    metric="path_gain",
                                    min_val_db=min_gain_db,
                                    max_val_db=max_gain_db,
                                    min_dist=min_dist,
                                    max_dist=max_dist)
    ue_pos = tf.squeeze(ue_pos)
    

if __name__ == '__main__':
    main()

