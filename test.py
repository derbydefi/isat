import jax.numpy as jnp
from isat import ISAT, SystemDynamics, PCA


# Define a simple system dynamics function for demonstration
def simple_system_dynamics(y, t, params):
    k1, k2 = params
    dydt = -k1 * y + k2 * (1 - y)
    return dydt

# Initialize system dynamics
params = jnp.array([0.1, 0.2])  # Example parameters
system_dynamics = SystemDynamics(simple_system_dynamics, params)

# Optional: Define dimensionality reduction function using PCA
dim_reduction_pca = PCA(n_components=2)

# Initialize ISAT with system dynamics, absolute tolerance, and optional PCA for dimensionality reduction
isat_model = ISAT(system_dynamics=system_dynamics, atol=1e-6, dim_reduce_func=None) # pca not working right now :(

# Example use case of ISAT
y0 = jnp.array([0.5])  # Initial condition
t_span = (0, 10)       # Time span for the query
params = jnp.array([0.1, 0.2])  # Parameters for the system dynamics

# Query the ISAT model
approximated_state_vector, optional_gradients = isat_model.query(y0, t_span, params, return_gradients=True)

print("Approximated State Vector:", approximated_state_vector)
if optional_gradients is not None:
    print("Gradients:", optional_gradients)
