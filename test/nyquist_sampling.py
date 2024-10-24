import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

# Load your IMU data
imu_data = np.load(os.path.expanduser('~/PycharmProjects/imu2pose-sionna/data/imu_reading.npy'))
x_axis_data_imu = imu_data[:, 0]

# Replace NaN values in the original data
if np.isnan(x_axis_data_imu).any():
    mean_value = np.nanmean(x_axis_data_imu)  # Compute mean excluding NaNs
    x_axis_data_imu = np.nan_to_num(x_axis_data_imu, nan=mean_value)  # Replace NaNs with the mean

# Define sampling parameters
imu_sampling_rate = 60  # Original sampling rate in Hz
downsampling_factor = 3  # Downsample by factor of 2 for Nyquist rate
quantization_levels_imu = 2**3  # Using * bits for quantization

# Time vector for the original data
imu_time = np.linspace(0, len(x_axis_data_imu) / imu_sampling_rate, num=len(x_axis_data_imu))

# Downsample the data
downsampled_imu_time = imu_time[::downsampling_factor]
downsampled_imu_data = x_axis_data_imu[::downsampling_factor]

# Quantize the downsampled data
max_imu_val = np.max(downsampled_imu_data)
min_imu_val = np.min(downsampled_imu_data)
range_imu_val = max_imu_val - min_imu_val

quantized_imu_data = (np.round((downsampled_imu_data - min_imu_val) / range_imu_val * (quantization_levels_imu - 1))
                      / (quantization_levels_imu - 1) * range_imu_val + min_imu_val)

# Convert quantized values to binary strings for visualization
# Adjust for 3-bit representation as quantization_levels_imu is set to 8
num_bits_required = int(np.log2(quantization_levels_imu))
binary_labels = [format(int((x - min_imu_val) / (max_imu_val - min_imu_val) * (quantization_levels_imu - 1)),
                        '0{}b'.format(num_bits_required)) for x in quantized_imu_data]


# Reconstruct the signal from quantized values
reconstruct_func_imu = interp1d(downsampled_imu_time, quantized_imu_data, kind='linear', fill_value="extrapolate")
reconstructed_imu_signal = reconstruct_func_imu(imu_time)

# Calculate the reconstruction error (MSE)
reconstruction_error = mean_squared_error(x_axis_data_imu, reconstructed_imu_signal)

# Create subplots with adjusted size
fig, axs = plt.subplots(4, 1, figsize=(12, 9))

# Plot original signal
axs[0].plot(imu_time, x_axis_data_imu, label='Original Signal', color='blue')
axs[0].set_title('Original IMU Signal (X-axis)')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Acceleration')
axs[0].grid(True)
axs[0].legend()

# Plot downsampled signal
axs[1].plot(imu_time, x_axis_data_imu, label='Original Signal', alpha=0.5, color='blue')
axs[1].scatter(downsampled_imu_time, downsampled_imu_data, color='red', label='Downsampled Points')
axs[1].set_title('Downsampled IMU Signal')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Acceleration')
axs[1].grid(True)
axs[1].legend()

# Plot quantized signal with binary y-axis labels
axs[2].scatter(downsampled_imu_time, quantized_imu_data, color='orange', marker='x', label='Quantized Points', s=100)
axs[2].set_yticks(quantized_imu_data)
axs[2].set_yticklabels(binary_labels)
axs[2].set_title('Quantized IMU Signal with Binary Representation')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('Binary Value')
axs[2].grid(True)
axs[2].legend()

# Plot reconstructed signal
axs[3].plot(imu_time, x_axis_data_imu, label='Original Signal', alpha=0.5, color='blue')
axs[3].plot(imu_time, reconstructed_imu_signal, label='Reconstructed Signal', linestyle='--', color='purple')
axs[3].set_title('Reconstructed IMU Signal from Quantized Data')
axs[3].set_xlabel('Time (seconds)')
axs[3].set_ylabel('Acceleration')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()

# Print the reconstruction error
print(f"Reconstruction Error (MSE): {reconstruction_error}")
