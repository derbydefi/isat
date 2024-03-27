import jax
import jax.numpy as jnp
from jax import grad, jit, vmap #jit/vmap usage experimental for now
from jax.experimental.ode import odeint
from jax.scipy.linalg import schur
from scipy.spatial import KDTree  # Retained for non-differentiable parts
import numpy as np  # For interfacing with SciPy KDTree, if needed (would be nice to use jax equivalent)

#    ___  ________  ________  _________   
#   |\  \|\   ____\|\   __  \|\___   ___\ 
#   \ \  \ \  \___|\ \  \|\  \|___ \  \_| 
#    \ \  \ \_____  \ \   __  \   \ \  \  
#     \ \  \|____|\  \ \  \ \  \   \ \  \ 
#      \ \__\____\_\  \ \__\ \__\   \ \__\
#       \|__|\_________\|__|\|__|    \|__|
#           \|_________|                  
#
# python implementation by derby
# ISAT algorithm concept, research, and reference by Dr. John Hedergren (BYU PRISM)


# =========================
# ISAT Algorithm Core Functions
# =========================

class ISAT:
    """
    Implements the In-Situ Adaptive Tabulation (ISAT) algorithm for efficient storage and retrieval 
    of computed states in dynamic systems simulations.
    
    Attributes:
        records (list): A list of dictionaries, each storing a record for a computed state.
        kdtree (KDTree): A KDTree object for efficient nearest-neighbor search within the records.
        system_dynamics (SystemDynamics): An instance of SystemDynamics to handle the dynamics and sensitivities.
        atol (float): Absolute tolerance for the approximation accuracy.
    """
        
    def __init__(self, system_dynamics, atol=1e-6,dim_reduce_func=None):
        """
        Initializes the ISAT instance.
        
        Parameters:
            system_dynamics (function): The system dynamics function to be tabulated.
            atol (float, optional): The absolute tolerance used for the ellipsoid of accuracy. Defaults to 1e-6.
        """
        self.records = []  # Stores the record dictionaries
        self.kdtree = None  # Will hold the KDTree instance
        self.system_dynamics = system_dynamics
        self.atol = atol
        self.dim_reduce_func = dim_reduce_func  

    def update_kdtree(self):
        if self.records:
            points = [record['start'] for record in self.records]
            self.kdtree = KDTree(points)
        else:
            self.kdtree = None


    def isat_growth(self, new_pt, i_leaf):
        """
        Expands the Ellipsoid of Accuracy (EOA) for a given point that falls outside the current EOA.
        
        Parameters:
            new_pt (numpy.ndarray): The new point that triggered the growth.
            i_leaf (int): The index of the leaf node in the KDTree corresponding to the nearest record.
        """
        print("growing EOA!")
        # Extract the record for the leaf index
        record = self.records[i_leaf]

        # Retrieve EOA and system dynamics parameters from the record
        A = jnp.array(record['eoa'])
        n = A.shape[0]
        atol_sq = self.atol ** 2

        # Compute the Schur decomposition of A
        Q, S = schur(A, output='real')
        e_vals = jnp.diag(S)



        # Compute the transformation T1 based on the square root of eigenvalues
        T1 = jnp.diag(jnp.sqrt(e_vals / atol_sq)) @ Q.T
        if not isinstance(new_pt, jnp.ndarray):
            new_pt = jnp.array(new_pt)  # Convert to jnp array only if it's not already
        T1_new_pt = T1 @ new_pt #make sure new_pt is dealt with properly here
        T1_new_pt_mag = jnp.linalg.norm(T1_new_pt, 2)
        T1_new_pt_dir = T1_new_pt / T1_new_pt_mag

        # Compute the expansion matrix E
        E = jnp.eye(n) - jnp.outer(T1_new_pt_dir, T1_new_pt_dir)
        E_vecs, E_vals = jnp.linalg.eig(E)
        T2 = jnp.linalg.inv(E_vecs)

        # Compute the M2 matrix for adjustment
        M2 = jnp.eye(n)
        M2 = M2.at[jnp.argmin(jnp.abs(jnp.diag(E_vals))), jnp.argmin(jnp.abs(jnp.diag(E_vals)))].set(1 / T1_new_pt_mag ** 2)

        # Calculate the expanded A
        A_expanded = atol_sq * (T1.T @ T2.T @ M2 @ T2 @ T1)

        record['eoa'] = np.asarray(A_expanded)  # convert back to np?

        # No need to update the KDTree here since the starting point hasn't changed
        # But if any logic requires updating the KDTree, call self.update_kdtree()

    def isat_add(self, y0, t_span=None, p=None, y=None, A=None,num_points=100):
        """
        Adds a new record to the ISAT database. This method is called when a new state vector and its sensitivity matrix need to be tabulated, which were obtained either through direct computation or provided by the user. 
        If dimensionality reduction is applied, it transforms the initial state before adding. Gradients of the system dynamics with respect to initial conditions and parameters are computed if supported by the system dynamics function.

        Parameters:
            y0 (numpy.ndarray): Initial condition, possibly transformed by a dimensionality reduction function if provided.
            t_span (tuple, optional): Time span for integration, required if `y` and `A` are not provided.
            p (numpy.ndarray, optional): Parameters for system dynamics, required if `y` and `A` are not provided.
            y (numpy.ndarray, optional): Computed state vector. It is computed from `y0` and `p` if not provided.
            A (numpy.ndarray, optional): Sensitivity matrix. It is computed from `y0` and `p` if not provided.
            num_points (int, optional): Number of points for numerical integration (default: 100).

        Returns:
            None
        """

        if self.dim_reduce_func is not None:
            # Apply dimensionality reduction if a function is provided
            y0 = self.dim_reduce_func(y0)

    
        if y is None or A is None:
            # Validate that t_span and p are provided
            if t_span is None or p is None:
                raise ValueError("t_span and p are required when y or A are not provided.")
            # Run the simulation to get y and A
            y, A = self.isat_integrate_and_sensitivities(y0, t_span, self.atol,num_points)
        gradients = self.system_dynamics.compute_gradients(y0, p) if hasattr(self.system_dynamics, 'compute_gradients') else None

        # Proceed with the existing logic of isat_add to calculate EOA and add the record
        assert isinstance(A, jnp.ndarray), "Expected A to be jnp.ndarray."
        U, S, V = jnp.linalg.svd(A)
        sigmas = jnp.maximum(self.atol**2 / 2, S)  # Thresholding the singular values
        #print("u,s,v,sigmas:",U,S,V,sigmas)
        modified_sensitivity_matrix = U @ jnp.diag(jnp.sqrt(sigmas)) @ V.T
        eoa = modified_sensitivity_matrix.T @ modified_sensitivity_matrix
        eoa_radius = jnp.sqrt(jnp.max(jnp.linalg.eigvals(eoa).real))        # Create and add the new record
        
        new_record = {
            'start': np.asarray(y0),
            'finish': np.asarray(y),
            'sensitivity': np.asarray(A),
            'eoa': np.asarray(eoa),
            'eoa_radius': float(eoa_radius),  # Ensure eoa_radius is a native Python float
            'gradient': gradients,  # Assuming gradients are already appropriately handled
            'accessed': 0
        }

        self.records.append(new_record)
        self.update_kdtree()

    # Example modification in isat_integrate_and_sensitivities or a dedicated method
    def compute_and_cache_gradients(self, y0, params, record_index):
        """
        Computes and caches the gradients of the system dynamics with respect to initial conditions
        and parameters for a specific record. This method leverages JAX for automatic differentiation,
        assuming the system dynamics function is compatible.

        Parameters:
            y0 (jnp.array): Initial state vector.
            params (jnp.array): System parameters.
            record_index (int): Index of the record where gradients are to be cached.

        Returns:
            None
        """
        grad_func = grad(self.system_dynamics.system_dynamics_func, argnums=(0, 1))
        gradients = grad_func(y0, params)
        self.records[record_index]['gradient'] = gradients


    def calculate_error(self, approximation, actual):
        """
        Calculates the error between the approximation and the actual values.
        
        Parameters:
            approximation (numpy.ndarray): The approximation vector.
            actual (numpy.ndarray): The actual state vector.
            
        Returns:
            float: The calculated error.
        """
        error = jnp.linalg.norm(approximation - actual) - self.atol
        return error
    
    def check_use_existing_approximation(self, i_leaf, y0):
        """
        Checks if the existing approximation at the leaf node can be used for the new query point.
        
        Parameters:
            i_leaf (int): Index of the leaf node in the records.
            y0 (numpy.ndarray): The new query point.
            
        Returns:
            numpy.ndarray or None: The approximation vector if it meets accuracy requirements, otherwise None.
        """
        record = self.records[i_leaf]
        approximation = record['finish']
        error = self.calculate_error(approximation, y0)  # Adjusted to compare relevant quantities
        
        if error < self.atol:
            return approximation
        else:
            return None
        


    def update_grow_approximation(self, i_leaf, y0, t_span, p):
        """
        Updates or grows the approximation for a given query point outside the initial EOA but close enough to include after expansion.

        Parameters:
            i_leaf (int): Index of the leaf in the records.
            y0 (numpy.ndarray): The new query point.
            t_span (tuple): The time span for the integration.
            p (numpy.ndarray): Parameters for the system dynamics.
        """
        record = self.records[i_leaf]

        # Convert record data to JAX arrays for computation, if necessary
        start_jax = jnp.array(record['start'])
        eoa_jax = jnp.array(record['eoa'])

        # Use system dynamics to integrate and calculate sensitivities
        y_final, sensitivities = self.isat_integrate_and_sensitivities(jnp.array(y0), t_span, self.atol)

        # Calculate the distance from the new point to the starting point of the record
        dy = y_final - start_jax
        eps_sq = jnp.dot(jnp.dot(dy.T, eoa_jax), dy)
        
        if eps_sq > 1:
            # If the new point is outside the current EOA, grow it using isat_growth
            self.isat_growth(np.array(y_final), i_leaf)  # Convert y_final to np array for isat_growth

        # Convert y_final and sensitivities back to np arrays before updating the record
        record['finish'] = np.array(y_final)
        record['sensitivity'] = np.array(sensitivities)


    #@jax.jit #will jit work here?
    def isat_integrate_and_sensitivities(self, y0, t_span, atol,num_points=100):
        """
        Integrates the system dynamics from an initial state `y0` over a time span `t_span` and
        computes the sensitivities of the final state with respect to the initial conditions.
        This method uses JAX's odeint for integration and automatic differentiation for sensitivity analysis.

        Parameters:
            y0 (numpy.ndarray): Initial condition for integration.
            t_span (tuple): Time span for integration.
            atol (float): Absolute tolerance for the integration.
            num_points (int): Number of points for integration (default: 100).

        Returns:
            tuple: A tuple (y_final, sensitivities) containing the final state and the computed sensitivities.
        """
        # Integrate the system dynamics over the given time span
        y_final = self.system_dynamics.isat_integrate(y0, t_span, atol,num_points)

        # Calculate sensitivities for the new final state
        _, sensitivities = self.system_dynamics.isat_sensitivities(y0, t_span, atol,num_points)

        return y_final, sensitivities

    
    def query(self, y0, t_span, params, return_gradients=False, use_existing=True, num_points=100):
    """
    Queries the ISAT database for an approximation of the system state based on the initial conditions, 
    optionally computing and returning the gradients of the system dynamics with respect to the initial 
    conditions and parameters. If an appropriate approximation does not exist in the database, 
    the method can compute a new state, add it to the database for future queries, and optionally 
    update an existing approximation.

    Parameters:
        y0 (numpy.ndarray): The initial conditions for the system simulation. This serves as the key for querying the ISAT database.
        t_span (tuple): A tuple representing the start and end times of the simulation, used to define the integration interval.
        params (numpy.ndarray): The parameters of the system dynamics function, which influence the behavior of the system being simulated.
        return_gradients (bool, optional): A flag indicating whether the gradients of the system state with respect to the initial conditions and parameters should be computed and returned. Defaults to False.
        use_existing (bool, optional): A flag that allows the method to use an existing approximation from the database if it meets the accuracy requirements. Defaults to True.
        num_points (int, optional): The number of points to use for numerical integration when computing a new state or updating an approximation. Defaults to 100.

    Returns:
        tuple: A tuple containing two elements:
            - approximated_state_vector (numpy.ndarray): The approximated state vector of the system at the end of the integration interval. This is either retrieved from the database, computed anew, or the result of updating an existing approximation.
            - optional_gradients (tuple or None): If `return_gradients` is True, this is a tuple containing the gradients of the system state with respect to the initial conditions and parameters. Otherwise, None.

    Example:
        # Define initial conditions, time span, and parameters for the query
        y0 = np.array([0.1, 0.2, 0.3])
        t_span = (0, 10)
        params = np.array([1.0, 0.5])

        # Query the ISAT database for an approximation, requesting gradients
        approximated_state, gradients = isat_instance.query(y0, t_span, params, return_gradients=True)

        # Use the approximated state and gradients for further analysis or visualization
    """
        # Initialize default return values
        approximated_state_vector = None
        optional_gradients = None

        if not self.records:
            print("No existing records. Adding a new one.")
            y_actual, sensitivities = self.isat_integrate_and_sensitivities(y0, t_span, self.atol, num_points=num_points)
            self.isat_add(y0, t_span, params, y_actual, sensitivities, num_points=num_points)
            approximated_state_vector = y_actual
            if return_gradients and hasattr(self.system_dynamics, 'compute_gradients'):
                optional_gradients = self.system_dynamics.compute_gradients(y0, params)
        else:
            self.update_kdtree()
            distance, i_leaf = self.kdtree.query(y0)
            record = self.records[i_leaf]

            if use_existing:
                existing_approximation = self.check_use_existing_approximation(i_leaf, y0)
                if existing_approximation is not None:
                    print("Using existing approximation.")
                    record['accessed'] += 1
                    approximated_state_vector = existing_approximation
                    if return_gradients and 'gradient' in record:
                        optional_gradients = record['gradient']
            
            if approximated_state_vector is None:
                y_actual, sensitivities = self.isat_integrate_and_sensitivities(y0, t_span, self.atol, num_points=num_points)
                error = self.calculate_error(y_actual, record['finish'])
                # Check if the EOA needs to be grown before adding a new record
                if self.should_grow_eoa(record, distance, error):
                    print("Growing EOA for existing record.")
                    self.isat_growth(y_actual, i_leaf)
                    approximated_state_vector = record['finish']  # Use the finish state from the record
                    if return_gradients and 'gradient' in record:
                        optional_gradients = record['gradient']
                else:
                    print("Adding new record.")
                    self.isat_add(y0, t_span, params, y_actual, sensitivities, num_points=num_points)
                    approximated_state_vector = y_actual
                    if return_gradients and hasattr(self.system_dynamics, 'compute_gradients'):
                        optional_gradients = self.system_dynamics.compute_gradients(y0, params)
        
        return approximated_state_vector, optional_gradients



    def should_grow_eoa(self, record, distance, error, eps_sq=None):
        """
        Determines whether the EOA should be expanded based on several criteria including error tolerance,
        access frequency, predictive analysis of query trends, and proximity of the query point to the record.

        Parameters:
            record (dict): The ISAT record associated with the nearest leaf.
            distance (float): The Euclidean distance from the query point to the nearest record.
            error (float): The computed error between the ISAT approximation and actual outcome for the new query.
            eps_sq (float, optional): Squared distance in transformed space, indicating how far the new point is from the EOA center.

        Returns:
            bool: True if the EOA should be grown, False otherwise.
        """
        # Define thresholds based on the scale of your data and domain-specific considerations
        distance_threshold = 0.1  # Adjust based on the scale of the data
        access_threshold = 5  # Adjust based on how often a record is considered "frequently accessed"
        eps_sq_threshold = 1.0  # Adjust based on the scale of transformed space distances

        # Grow the EOA if the error is within tolerance and the query point is close enough, frequently accessed, or within a transformed space proximity
        if error <= self.atol:
            if distance <= distance_threshold:
                print("Close enough to existing record, considering EOA growth.")
                return True
            elif record['accessed'] > access_threshold:
                print("Frequently accessed record, considering EOA growth.")
                return True
            elif eps_sq is not None and eps_sq <= eps_sq_threshold:
                print("Within transformed space proximity, considering EOA growth.")
                return True
        
        # Default to not growing the EOA if none of the conditions are met
        print("Not growing EOA.")
        return False

# =========================
# System Dynamics and Sensitivity Analysis
# =========================

class SystemDynamics:
    """
    Encapsulates the system dynamics and sensitivity analysis for use within the ISAT algorithm,
    using autograd for automatic differentiation of Jacobians and sensitivities.
    """
    def __init__(self, system_dynamics_func, params=None):
        """
        Initializes the system dynamics with autograd-based differentiation.

        Parameters:
        - system_dynamics_func (function): The system dynamics function f(t, y, params) using autograd.numpy.
        - parameters (numpy.ndarray): Parameters for the system dynamics.
        """
        self.system_dynamics_func = system_dynamics_func
        self.params = params

    def compute_gradients(self, y0, params):
        """
        Computes gradients of the system dynamics with respect to both the initial condition y0 and parameters.
        This function uses JAX for automatic differentiation, facilitating gradient computation for optimization
        or sensitivity analysis purposes.

        Parameters:
            y0 (jnp.array): The initial conditions of the system.
            params (jnp.array): Parameters of the system dynamics.

        Returns:
            tuple: A tuple containing gradients with respect to y0 and parameters. 
        """
        def final_state(y0, params):
            t_span = jnp.linspace(0, 10, 100)  # Define time span for integration
            solution = odeint(lambda y, t: self.system_dynamics_func(y, t, params), y0, t_span)
            return jnp.linalg.norm(solution[-1]) 
        
        grad_y0 = jax.grad(final_state, argnums=0)(y0, params)
        grad_params = jax.grad(final_state, argnums=1)(y0, params)
        
        return grad_y0, grad_params

    #@jax.jit #will jit work here?    
    def system_dynamics(self, t, y):
        """
        Defines the system dynamics function. This function should use JAX-compatible numpy (jnp) for
        all operations to ensure compatibility with automatic differentiation.

        Parameters:
            t (float): The current time point.
            y (jnp.array): The current state of the system.

        Returns:
            jnp.array: The time derivative of the state.
        """
        return self.system_dynamics_func(t, y, self.params)

    def jacobian_matrix(self, y, t):
        """
        Computes the Jacobian matrix of the system dynamics with respect to the state vector y at a given time t,
        using JAX's automatic differentiation capabilities.

        Parameters:
            y (jnp.array): The current state vector of the system.
            t (float): The current time.

        Returns:
            jnp.array: The Jacobian matrix of the system dynamics.
        """
        # Use JAX's automatic differentiation to compute the Jacobian
        # Note: Ensure `system_dynamics` is compatible with JAX and uses jnp instead of np
        J = jax.jacfwd(self.system_dynamics, argnums=1)(t, y)
        return J

    #@jax.jit
    def solve_augmented_system(self, y0, t_span, atol=1e-6, rtol=1e-3,num_points=100):
        """
        Solves the augmented system for sensitivities using JAX.
        """
        y0_jax = jnp.array(y0)
        t_span_jax = jnp.linspace(t_span[0], t_span[1], num_points) #added num_points instead of harcoded 100
        n = y0_jax.shape[0]  # Dimension of the system

        # Initialize sensitivity part of the augmented system with identity matrix
        S_initial = jnp.eye(n).flatten()
        y_augmented_initial = jnp.concatenate([y0_jax, S_initial])

        #print(f"Initial state vector (y0_jax): {y0_jax.shape}")
        #print(f"Initial sensitivity matrix (S_initial): {jnp.eye(n).shape}")
        #print(f"Initial augmented state (y_augmented_initial): {y_augmented_initial.shape}")

        def augmented_system(y_augmented, t):
            y = y_augmented[:n]
            S = y_augmented[n:].reshape((n, n))
            dydt = self.system_dynamics(t, y)
            J = self.jacobian_matrix(y, t)
            dSdt = jnp.matmul(J, S)

            # Print shapes of involved matrices
            #print(f"At time {t}, state vector (y): {y.shape}, Jacobian (J): {J.shape}, Sensitivity (S): {S.shape}, dSdt: {dSdt.shape}")
            dydt_1d = jnp.atleast_1d(dydt)
            dSdt_1d = jnp.atleast_1d(dSdt.flatten())
            return jnp.concatenate([dydt_1d, dSdt_1d])

        solution = odeint(augmented_system, y_augmented_initial, t_span_jax, rtol=rtol, atol=atol)
        Y = solution[-1][:n]
        S_final = solution[-1][n:].reshape((n, n))

        # Print the final shapes
        #print(f"Final state vector (Y): {Y.shape}")
        #print(f"Final sensitivity matrix (S_final): {S_final.shape}")

        return Y, S_final

    #@jax.jit # will jit wwork here?
    def isat_sensitivities(self, y0, t_span, atol=1e-6, rtol=1e-3,num_points=100):
        """
        Calculates the system state sensitivities with respect to initial conditions using the augmented system approach.
        
        Parameters:
            y0 (numpy.ndarray): Initial state vector.
            t_span (tuple): Time span for integration.
            atol (float): Absolute tolerance for integration.
            rtol (float): Relative tolerance for integration.
        
        Returns:
            tuple: Final state vector and sensitivities matrix.
        """
        # Leverage the unified method for computing sensitivities
        Y, S_final = self.solve_augmented_system(y0, t_span, atol, rtol,num_points)
        return Y, S_final

    #@jax.jit #will jit work here?
    def isat_integrate(self, y0, t_span, atol=1e-6, rtol=1e-3,num_points=100):
        """
        Integrates the system dynamics over the specified time span using JAX's odeint.
        
        Parameters:
            y0 (jnp.array): Initial state vector.
            t_span (tuple): Time span for integration.
            atol (float): Absolute tolerance for integration.
            rtol (float): Relative tolerance for integration.
            
        Returns:
            jnp.array: State vector at the end of the integration period.
        """
        # Convert inputs to JAX arrays if they aren't already
        y0_jax = jnp.array(y0)
        t_span_jax = jnp.linspace(t_span[0], t_span[1], num_points)  

        # Define the system dynamics for use with JAX's odeint
        def system_dynamics_wrapper(y, t):
            return self.system_dynamics(t, y)

        # Integrate the system dynamics over the time span
        solution = odeint(system_dynamics_wrapper, y0_jax, t_span_jax, rtol=rtol, atol=atol)

        # Return the state vector at the end of the integration period
        return solution[-1]

# =========================
# Utilities
# =========================

class PCA:
    """
    Principal Component Analysis (PCA) using JAX for efficient computation.
    """
    def __init__(self, n_components=2):
        """
        Initializes the PCA instance with the desired number of components.
        
        Parameters:
            n_components (int): The number of principal components to compute.
        """
        self.n_components = n_components
    
    #@jit
    def reduce(self, data):
        """
        Applies PCA to reduce the dimensionality of the given data.
        
        Parameters:
            data (jax.numpy.ndarray): The data to be reduced in dimensionality.
            
        Returns:
            jax.numpy.ndarray: The data projected into the space defined by the top principal components.
        """
        # Center the data
        data_centered = data - jnp.mean(data, axis=0)
        
        # Compute covariance matrix
        cov_matrix = jnp.cov(data_centered, rowvar=False)
        
        # Eigen decomposition
        eigen_values, eigen_vectors = jnp.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors
        idx = jnp.argsort(eigen_values)[::-1]
        sorted_eigen_vectors = eigen_vectors[:, idx]
        
        # Select the top n_components eigenvectors
        principal_components = sorted_eigen_vectors[:, :self.n_components]
        
        # Project the data onto the principal components
        transformed_data = jnp.dot(data_centered, principal_components)
        
        return transformed_data
    def __call__(self, data):
        return self.reduce(data)


