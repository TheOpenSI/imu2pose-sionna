import sionna
import sionna_vispy
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, watt_to_dbm
from sionna.mimo.precoding import normalize_precoding_power, grid_of_beams_dft

def test_coverage_map():
    scene = load_scene()  # Load empty scene

    # Configure antenna arrays for all transmitters and receivers
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,  # relative to wavelength
                                 horizontal_spacing=0.5,  # relative to wavelength
                                 pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array

    # Define and add a first transmitter to the scene
    tx0 = Transmitter(name='tx0',
                      position=[150, -100, 20],
                      orientation=[np.pi * 5 / 6, 0, 0],
                      power_dbm=44)
    scene.add(tx0)

    # Compute coverage map
    cm = scene.coverage_map(max_depth=5,  # Maximum number of ray scene interactions
                            num_samples=int(10e6),  # If you increase: less noise, but more memory required
                            cm_cell_size=(5, 5),  # Resolution of the coverage map
                            cm_center=[0, 0, 0],  # Center of the coverage map
                            cm_size=[400, 400],  # Total size of the coverage map
                            cm_orientation=[0, 0, 0])  # Orientation of the coverage map, e.g., could be also vertical
    print(cm)
    # Visualize path gain
    cm.show(metric="path_gain")

    # Visualize received signal strength (RSS)
    cm.show(metric="rss")

    # Visulaize SINR
    cm.show(metric="sinr")
    plt.show()


if __name__ == '__main__':
    test_coverage_map()