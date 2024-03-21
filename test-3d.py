import numpy as np
import matplotlib.pyplot as plt
from isat import ISAT
from mpl_toolkits.mplot3d import Axes3D

# Define a nonlinear target function
def target_function(input_array):
    x, y = input_array
    return np.sin(np.sqrt(x**2 + y**2)) * np.exp(-0.1 * np.sqrt(x**2 + y**2))

# Parameters
error_tolerance = 0.1  # Error tolerance for ISAT
training_samples = 100  # Number of training samples
test_samples = 100  # Number of test samples for demonstration

# Initialize the ISAT model
isat_model = ISAT(error_tolerance=error_tolerance, output_function=target_function)

# Training
x_train, y_train = np.meshgrid(np.linspace(-10, 10, training_samples), np.linspace(-10, 10, training_samples))
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        input_array = np.array([x_train[i, j], y_train[i, j]])
        z = target_function(input_array)
        isat_model.add_approximation(input_array, np.array([z]))

# Testing and plotting
x_test, y_test = np.meshgrid(np.linspace(-10, 10, test_samples), np.linspace(-10, 10, test_samples))
z_approx = np.zeros(x_test.shape)
z_actual = np.zeros(x_test.shape)
for i in range(x_test.shape[0]):
    for j in range(y_test.shape[1]):
        input_array = np.array([x_test[i, j], y_test[i, j]])
        z_actual[i, j] = target_function(input_array)
        # This line should use the ISAT model to get the approximation
        z_approx[i, j] = isat_model.query_approximation(input_array).item()  # Assume the ISAT model returns a numpy array and extract the scalar


# Calculate the error surface (absolute difference between actual and approximation)
z_error = np.abs(z_actual - z_approx)

# Plotting
fig = plt.figure(figsize=(18, 6))

# Plot actual surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x_test, y_test, z_actual, cmap='viridis', alpha=0.7)
ax1.set_title('Actual Function Surface')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

# Plot approximated surface
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x_test, y_test, z_approx, cmap='inferno', alpha=0.7)
ax2.set_title('ISAT Approximation Surface')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

# Plot error surface
ax3 = fig.add_subplot(133, projection='3d')
surf = ax3.plot_surface(x_test, y_test, z_error, cmap='coolwarm')
ax3.set_title('Error Surface (Actual - Approximation)')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Error')

# Add a color bar to the error surface plot to quantify the error
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)

plt.show()