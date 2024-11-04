import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def pre_processing_imu(input_data):
    """
        Pre-processes the IMU data by interpolating NaN values.

        Parameters:
            input_data (numpy array): 2D array of IMU sensor readings with arbitrary number of features.

        Returns:
            input_data (numpy array): Pre-processed IMU data with NaN values interpolated.
        """
    # Interpolate to handle NaN values for all features
    for i in range(input_data.shape[1]):  # Loop over features (columns)
        nan_indices = np.isnan(input_data[:, i])
        if np.any(nan_indices):
            valid_indices = ~nan_indices
            input_data[nan_indices, i] = np.interp(
                np.flatnonzero(nan_indices),
                np.flatnonzero(valid_indices),
                input_data[valid_indices, i]
            )

    return input_data


def visualize_imu_data(imu_data, num_features=3):
    """
    Visualizes the IMU data with subplots for each feature (axis).

    Parameters:
        imu_data (numpy array): 2D array of IMU sensor readings with arbitrary number of features.
        num_features (int): Number of features to visualize.
    """
    # Create subplots for each feature
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 2 * num_features))

    time = np.arange(imu_data.shape[0])  # Create a time index based on the number of samples

    # Plot the data for each feature
    for i in range(num_features):
        axs[i].plot(time, imu_data[:, i])
        axs[i].set_title(f'Feature {i + 1} Data')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel(f'Feature {i + 1} Value')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def imu_to_binary(imu_data, quantization_level):
    """
    Transforms IMU sensor data into a binary sequence and handles NaN values with interpolation.

    Parameters:
        imu_data (numpy array): 2D array of IMU sensor readings.
        quantization_level (int): Number of quantization levels.

    Returns:
        binary_sequence (str): The binary representation of quantized IMU data.
    """
    # Normalize the IMU data to the range [0, 1]
    data_min = np.min(imu_data)
    data_max = np.max(imu_data)

    if data_min == data_max:
        normalized_data = np.zeros_like(imu_data)
    else:
        normalized_data = (imu_data - data_min) / (data_max - data_min)

    # Quantize the data
    quantized_data = np.floor(normalized_data * (quantization_level - 1)).astype(int)

    # Convert the quantized data into a binary sequence
    bits_per_value = int(np.ceil(np.log2(quantization_level)))
    bit_array = np.array([list(format(value, f'0{bits_per_value}b')) for value in quantized_data.flatten()], dtype=int)

    # Reshape the bit array into the appropriate shape (number of values * bits per value)
    bit_array = bit_array.flatten()

    return bit_array


def binary_to_imu(binary_sequence, quantization_level, original_shape, data_min, data_max):
    """
    Recovers the original data from a binary sequence (with quantization error).

    Parameters:
        binary_sequence (str): The binary representation of quantized IMU data.
        quantization_level (int): Number of quantization levels.
        original_shape (tuple): Shape of the original data.
        data_min (float): Minimum value of the original data.
        data_max (float): Maximum value of the original data.

    Returns:
        recovered_data (numpy array): The recovered data with quantization error.
    """
    # Determine the number of bits used per quantized value
    bits_per_value = int(np.ceil(np.log2(quantization_level)))
    # print('bits_per_value: {}'.format(bits_per_value))

    # Ensure the binary sequence is a string, or convert a numpy array of bits to a string
    if isinstance(binary_sequence, np.ndarray):
        # Join the bits into a single binary string
        binary_sequence = ''.join(binary_sequence.astype(str))

    # Adjust binary sequence length to match the total size required
    total_elements = np.prod(original_shape)
    required_bits = total_elements * bits_per_value

    # Ensure the binary sequence has enough bits
    if len(binary_sequence) < required_bits:
        raise ValueError(
            f"Not enough bits in the binary sequence: {len(binary_sequence)} available, {required_bits} required.")

    binary_sequence = binary_sequence[:required_bits]  # Trim or pad if necessary

    # Split the binary sequence into chunks of size bits_per_value
    quantized_values = [
        int(binary_sequence[i * bits_per_value:(i + 1) * bits_per_value], 2)
        for i in range(len(binary_sequence) // bits_per_value)
    ]

    # Ensure the quantized values can be reshaped into the original shape
    if len(quantized_values) != total_elements:
        raise ValueError(
            f"Number of quantized values ({len(quantized_values)}) does not match the required number of elements ({total_elements}).")

    # Reshape the quantized values back into the original shape
    quantized_array = np.array(quantized_values).reshape(original_shape)

    # Rescale the quantized data back to the original range [data_min, data_max]
    normalized_recovered_data = quantized_array / (quantization_level - 1)
    recovered_data = normalized_recovered_data * (data_max - data_min) + data_min

    return recovered_data

def numpy_to_tensorflow_source(bit_array, shape):
    """
    Converts a numpy array of binary bits into a TensorFlow 2D tensor with a specified shape.

    Parameters:
        bit_array (numpy array): The numpy array containing binary bits (0 or 1).
        shape (tuple): The target shape for the TensorFlow tensor (e.g., (n, m)).

    Returns:
        tf_tensor (tf.Tensor): A TensorFlow tensor with the desired shape.
    """
    # Step 1: Convert the numpy array of bits to a TensorFlow tensor
    tf_tensor = tf.convert_to_tensor(bit_array, dtype=tf.float32)

    # Step 2: Reshape the tensor to the desired shape
    tf_tensor = tf.reshape(tf_tensor, shape)

    return tf_tensor


def test_imu_functions():
    # Load IMU data
    path = '/Users/hieu/datasets/DIP_IMU_and_Others/processed_test.npz'
    imu = np.load(path)['imu']
    len_imu = imu.shape[0]
    print('imu.shape: {}'.format(imu.shape))

    batch_size = 5000
    min_val = -1.0
    max_val = 1.0

    # Assume arbitrary feature length, modify this line to adapt
    num_features = imu.shape[-1]

    # Reshape to (num_samples, num_features) if needed
    imu_data = imu.reshape((len_imu, num_features))

    # Pre-process IMU data
    imu_data = imu_data[:batch_size]
    imu_data = pre_processing_imu(imu_data)
    print('imu_data shape after preprocessing: {}'.format(imu_data.shape))

    # Visualize IMU data
    visualize_imu_data(imu_data)

    # Specify quantization level
    quantization_level = 2 ** 7

    # Get binary sequence
    binary_sequence = imu_to_binary(imu_data, quantization_level)

    # Show first 100 bits of the binary sequence
    print(binary_sequence[:100])
    print('Binary sequence length: {}'.format(len(binary_sequence)))

    # Recover data
    recovered_data = binary_to_imu(binary_sequence, quantization_level,
                                   imu_data.shape, min_val, max_val
                                   )
    print('recovered_data shape: {}'.format(recovered_data.shape))
    mse = np.mean((imu_data - recovered_data) ** 2)
    print('MSE between original and recovered data: {}'.format(mse))

    visualize_imu_data(recovered_data)


if __name__ == '__main__':
    test_imu_functions()