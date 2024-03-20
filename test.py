import numpy as np
import matplotlib.pyplot as plt
from isat import ISAT


# Define a nonlinear target function
def target_function(x):
    return np.sin(x) * np.log(x + 1)

# Generate sample input data and their corresponding outputs
x_train = np.linspace(1, 10, 20)  # Training inputs
y_train = target_function(x_train)  # Training outputs

x_test = np.linspace(1, 10, 100)  # Testing inputs for demonstration
y_test = target_function(x_test)  # Actual outputs for testing inputs


# Initialize the ISAT model with a high error tolerance for demonstration
isat_model = ISAT(error_tolerance=0.1, output_function=target_function)

# Simulate "training" by adding initial approximations from the training dataset
for x, y in zip(x_train, y_train):
    isat_model.add_approximation(np.array([x]), np.array([y]))


# Placeholder for ISAT approximations
y_approx = []

# For demonstration, let's just collect actual target function values
# In practice, you would query the ISAT model for approximations here
for x in x_test:
    approx = isat_model.output_function(np.array([x]))
    y_approx.append(approx)

# Plot the actual vs. approximated values for demonstration
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label='Actual Target Function', linestyle='--')
plt.scatter(x_train, y_train, label='Training Data', color='red')
plt.plot(x_test, y_approx, label='ISAT Approximations', color='green')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ISAT Approximation vs. Actual Target Function')
plt.show()
