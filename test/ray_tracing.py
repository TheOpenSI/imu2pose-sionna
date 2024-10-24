import os
import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sionna_vispy

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHReceiver, PUSCHTransmitter
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement


if __name__ == '__main__':
    scene = load_scene(sionna.rt.scene.munich)  # Try also sionna.rt.scene.etoile
    my_cam = Camera("my_cam", position=[-250, 250, 150], look_at=[-15, 30, 28])
    scene.add(my_cam)

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
                     position=[28.5, 21, 10])

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a receiver
    rx = Receiver(name="rx",
                  position=[45, 90, 1.5],
                  orientation=[0, 0, 0])

    # Add receiver instance to scene
    scene.add(rx)

    tx.look_at(rx)  # Transmitter points towards receiver

    scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials

    scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

    # Select an example object from the scene
    so = scene.get("Altes_Rathaus-itu_marble")

    # Print name of assigned radio material for different frequenies
    for f in [3.5e9, 2.14e9]:  # Print for differrent frequencies
        scene.frequency = f
        print(f"\nRadioMaterial: {so.radio_material.name} @ {scene.frequency / 1e9:.2f}GHz")
        print("Conductivity:", so.radio_material.conductivity.numpy())
        print("Relative permittivity:", so.radio_material.relative_permittivity.numpy())
        print("Complex relative permittivity:", so.radio_material.complex_relative_permittivity.numpy())
        print("Relative permeability:", so.radio_material.relative_permeability.numpy())
        print("Scattering coefficient:", so.radio_material.scattering_coefficient.numpy())
        print("XPD coefficient:", so.radio_material.xpd_coefficient.numpy())

    # Compute propagation paths
    paths = scene.compute_paths(max_depth=5,
                                num_samples=1e6)  # Number of rays shot into directions defined
    # by a Fibonacci sphere , too few rays can
    # lead to missing paths

    # Visualize paths in the 3D preview
    # scene.render("my_cam", paths=paths, show_devices=True, show_paths=True)

    with sionna_vispy.patch():
        canvas = scene.preview(paths, show_devices=True, show_paths=True)

    canvas.show()
    canvas.app.run()
