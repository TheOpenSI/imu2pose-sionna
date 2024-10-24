import sionna
import sionna_vispy
from sionna.rt import PlanarArray, Transmitter, load_scene, visualize
import matplotlib.pyplot as plt
import tensorflow as tf


# load scene
scene = load_scene(sionna.rt.scene.floor_wall)
scene.render(camera="scene-cam-0")

# test radio device
scene.tx_array = PlanarArray(num_rows=4,
                              num_cols=2,
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="tr38901",
                              polarization="cross")

my_tx = Transmitter(name="my_tx",
                     position=(0,0,0),
                     orientation=(0,0,0))

scene.add(my_tx)
print(scene.tx_array.antenna.patterns)
tx_patterns = scene.tx_array.antenna.patterns
visualize(tx_patterns[0])
plt.show()

# rotation
position = scene.tx_array.positions
print('Current position:', position)
orientation = [3.14/2, 3.14/4, 3.14/9]
new_positions = tf.constant(orientation, dtype=tf.float32)
scene.tx_array.positions = new_positions
tx_patterns = scene.tx_array.antenna.patterns
visualize(tx_patterns[0])
plt.show()

with sionna_vispy.patch():
    scene.render(camera="scene-cam-0")
    canvas = scene.preview()
canvas.show()
canvas.app.run()
