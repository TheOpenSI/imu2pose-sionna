import os
import platform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os_name = platform.system()
if os_name == 'Linux':
    imu_dataset_path = os.path.expanduser('~/Data/datasets/DIP_IMU_and_Others/') 
else:
    imu_dataset_path = os.path.expanduser('~/datasets/DIP_IMU_and_Others/') 
    

def prepare_source_data(num_imu_frames, batch_size, n, quantization_level):
    """
    Perform quantization on original IMU data and transform source IMU data into source bits. 
    
    Args:
        - num_imu_frames (int): number of IMU data frames used as source data 
        - batch_size (int): batch size of a single forward through OFDM channel. This will be used 
        as input of the call() function in the CustomBinarySource class
        - n (int): number of data bits per OFDM resource grid
        - quantization_level (int): number of quantization levels used for quantizing IMU signals
        
    Return:
        - source_ofdm_bit [num_batches, batch_size, 1, 1, n]: source bits to be transmitted over OFDM channel
        - source_imu_quantized [num_batches, 204], source IMU data after quantization 
        - source_imu_original [num_batches, 204], source IMU data before quantization
    """
    # load IMU data
    path = imu_dataset_path +  'processed_test.npz'
    data = np.load(path)['imu']
    data = np.squeeze(data)
    data = pre_processing_imu(data)
    print('Original IMU shape: {}'.format(data.shape))
    
    # Get a fixed number of IMU samples [num_imu_frames, 204]
    data = data[:num_imu_frames]
    print('Source IMU shape: {}'.format(data.shape))

    # Quantization imu [num_imu_frames, 204] -> source_bits [num_imu_frames, 204, bits_per_value] 
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    source_bits = imu_to_binary(data, quantization_level, data_min, data_max)
    source_bits = np.asarray(source_bits)
    print('Quantized source IMU shape: {} ~ {} bits '.format(source_bits.shape, np.prod(source_bits.shape)))
     
    # Transform source_bits [num_imu_frames, 204, bits_per_value] into ofdm_bits [(num_imu_frames / batch_size) * (204 * bits_per_value / n), batch_size, 1, 1, n]
    bits_per_value = int(np.ceil(np.log2(quantization_level)))
    num_ofdm_rg_batches = (num_imu_frames / batch_size) * (204 * bits_per_value / n)
    num_ofdm_rg_batches = int(num_ofdm_rg_batches)  # 30
    num_source_bits_used = num_ofdm_rg_batches * batch_size * n  # 8352000
    print('Number of source bits used: {}'.format(num_source_bits_used))
    # trim the source_bits into a shorter sequence, so that we can transform it into batches of ofdm_bits
    source_bits = source_bits.flatten()[:num_source_bits_used]
    ofdm_bits = np.reshape(source_bits, [num_ofdm_rg_batches, batch_size, 1, 1, n])
    print('OFDM source bits shape: {}'.format(ofdm_bits.shape))
    
    # De-quantization bits [num_ofdm_rg_batches, batch_size, 1, 1, n] -> [num_imu_frames_used, 204, bits_per_value] 
    # and -> [num_imu_frames_used, 204], note that num_imu_frames_used will be shorter due to the bits <-> ofdm_bits conversation
    num_imu_frames_used = (num_ofdm_rg_batches * batch_size * n) / (204 * bits_per_value)
    num_imu_frames_used = int(num_imu_frames_used)  # 5848
    print('Number of IMU frames used: {}'.format(num_imu_frames_used))
    
    imu_quantized = binary_to_imu(ofdm_bits, quantization_level, (num_imu_frames_used, 204), data_min, data_max)
    imu_original = data[:num_imu_frames_used]

    return ofdm_bits, imu_quantized, imu_original

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

def imu_to_binary(imu_data, quantization_level, data_min, data_max):
    """
    Transforms IMU sensor data into a binary sequence.

    Parameters:
        imu_data (numpy array): 2D array of IMU sensor readings.
        quantization_level (int): Number of quantization levels.
        data_min (numpy array): 1D array of minimum values for each of the 204 features.
        data_max (numpy array): 1D array of maximum values for each of the 204 features.

    Returns:
        binary_sequence (numpy flatten array): The binary representation of quantized IMU data.
    """
    # Check if the data_min and data_max are valid
    if np.any(data_min == data_max):
        raise ValueError("IMU data has zero range for one or more features. All values are identical, so quantization is not possible for those features.")
    
    # Normalize the IMU data to the range [0, 1] for each feature dimension
    normalized_data = (imu_data - data_min) / (data_max - data_min)

    # Quantize the data
    quantized_data = np.floor(normalized_data * (quantization_level - 1)).astype(int)

    # Determine bits per value
    bits_per_value = int(np.ceil(np.log2(quantization_level)))

    # Vectorized binary conversion using bitwise operations
    binary_array = ((quantized_data[:, :, None] & (1 << np.arange(bits_per_value)[::-1])) > 0).astype(int)

    return binary_array


def binary_to_imu(binary_sequence, quantization_level, imu_shape, data_min, data_max):
    """
    Recovers the original data from a binary sequence (with quantization error).

    Parameters:
        binary_sequence (numpy array or str): The binary representation of quantized IMU data.
        quantization_level (int): Number of quantization levels.
        imu_shape (tuple): Shape of the IMU data (seq_len, 204).
        data_min (numpy array, optional): 1D array of minimum values for each of the 204 features.
        data_max (numpy array, optional): 1D array of maximum values for each of the 204 features.

    Returns:
        recovered_data (numpy array): The recovered data (seq_len, 204) with quantization error.
    """
    # Determine the number of bits used per quantized value
    bits_per_value = int(np.log2(quantization_level))

    # Ensure the binary sequence is a string, or convert a numpy array of bits to a string
    if isinstance(binary_sequence, np.ndarray):
        binary_sequence = ''.join(map(str, binary_sequence.flatten()))
        
    print('len binary_sequence: {}'.format(len(binary_sequence)))
    
    # if not np.all(np.isin(binary_sequence, [0, 1])):
    #     raise ValueError("Binary sequence must only contain 0s and 1s.")
    
   # Split the binary sequence into chunks of size bits_per_value and convert to integer
    try:
        quantized_values = [
            int(binary_sequence[i * bits_per_value:(i + 1) * bits_per_value], 2)
            for i in range(len(binary_sequence) // bits_per_value)
        ]
    except ValueError as e:
        raise ValueError(f"Invalid binary sequence chunk detected. Ensure that the binary sequence contains only '0' and '1'. Original error: {e}")
    
    total_elements = np.prod(imu_shape)
    print('len quantized_values: {}'.format(len(quantized_values)))
    quantized_values = quantized_values[:total_elements]  # Trim excess values

    # Reshape the quantized values back into the original shape
    quantized_values = np.array(quantized_values).reshape(imu_shape)
    
    # Check if the data_min and data_max are valid
    if np.any(data_min == data_max):
        raise ValueError("IMU data has zero range for one or more features. All values are identical, so rescaling is not possible for those features.")

    # Rescale the quantized data back to the original range per feature dimension
    recovered_data = quantized_values / (quantization_level - 1)
    recovered_data = recovered_data * (data_max - data_min) + data_min

    return recovered_data