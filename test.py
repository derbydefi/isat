import numpy as np
import matplotlib.pyplot as plt
from isat import ISAT

# Define a nonlinear target function
def target_function(x):
    return np.sin(x) * np.log(x + 1)


# Parameters
error_tolerance = 0.1  # Error tolerance for ISAT
training_samples = 10  # Number of training samples
test_samples = 10  # Number of test samples for demonstration

# Initialize the ISAT model
isat_model = ISAT(error_tolerance=error_tolerance, output_function=target_function)

# Training - Add initial approximations from the training dataset
x_train = np.linspace(1, 10, training_samples)
y_train = target_function(x_train)
for x, y in zip(x_train, y_train):
    isat_model.add_approximation(np.array([x]), np.array([y]))

# Testing - Use ISAT approximations for demonstration
x_test = np.linspace(1, 10, test_samples)
y_approx = []
for x in x_test:
    nearest_node = isat_model.find_nearest_approximation(np.array([x]))
    if nearest_node:
        # Use the approximation from the nearest node
        approx = nearest_node.data[1]
    else:
        # If no nearest node found, fallback to direct evaluation
        # This condition is simplified; in practice, you'd dynamically update the ISAT model here
        approx = target_function(np.array([x]))
        # Normally, you would update the ISAT model with this new data point
        # isat_model.insert_or_update_approximation(np.array([x]), np.array([approx]))
    y_approx.append(approx)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_test, target_function(x_test), label='Actual Target Function', linestyle='--')
plt.scatter(x_train, y_train, label='Training Data', color='red')
plt.plot(x_test, y_approx, label='ISAT Approximations', color='green')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ISAT Approximation vs. Actual Target Function')
plt.show()
