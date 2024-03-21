import numpy as np



class KDNode:
    def __init__(self, data=None, left=None, right=None, axis=0):
        self.data = data
        self.left = left
        self.right = right
        self.axis = axis  # The dimension/axis the node splits

class ISAT:
    def __init__(self,error_tolerance=1e-5, output_function=None):
        """Initializes the In-Situ Adaptive Tabulation (ISAT) model with a specified error tolerance. This tolerance determines when to update or expand the EOA. The model starts with no nodes and various counters set to zero."""
        self.root = None
        self.error_tolerance = error_tolerance
        self.output_function = output_function
        self.leaf_count = 0
        self.node_count = 0
        self.add_count = 0
        self.grow_count = 0
        self.retrieve_count = 0
        self.direct_eval_count = 0
        self.height = 0
        self.max_leaves = 0


    def add_approximation(self, input_data, output_data):
        """ Adds approximation to ISAT's kdtree """
        gradient_matrix = self.recalculate_gradient(input_data, output_data)
        eoa_matrix = np.eye(len(input_data))  # Assuming multi-dimensional input
        new_node = KDNode(data=(input_data, output_data, gradient_matrix, eoa_matrix))
        if self.root is None:
            self.root = new_node
        else:
            self._insert_kd_tree(self.root, new_node, 0)
    def query_approximation(self, input_data):
        nearest_node = self.find_nearest_approximation(input_data)
        actual_output = self.output_function(input_data)

        if nearest_node is not None:
            predicted_output = nearest_node.data[1]
            error = self.calculate_local_error(actual_output, predicted_output)
            
            if error <= self.error_tolerance:
                return predicted_output
            else:
                self.update_approximation(nearest_node, input_data, actual_output)
                return actual_output
        else:
            self.add_approximation(input_data, actual_output)
            return actual_output




    def update_approximation(self, node, new_input, new_output):
        input_data, output_data, gradient_matrix, eoa_matrix = node.data
        A_prime = self.calculate_jacobian(new_input)
        local_error = self.calculate_local_error(output_data, new_output)
        
        eoa_matrix_updated = self.update_eoa_matrix_advanced(gradient_matrix, local_error, self.error_tolerance)
        
        node.data = (input_data, new_output, A_prime, eoa_matrix_updated)



    def recalculate_gradient(self, input_data, output_data):
        """Calculates the gradient for n-dimensional input by applying a small perturbation to each dimension."""
        delta = 1e-5
        num_dims = input_data.shape[0]  # Assuming input_data is an np.array with shape (n_dims,)
        gradient = np.zeros(num_dims)
        for i in range(num_dims):
            perturbed_input = np.copy(input_data)
            perturbed_input[i] += delta
            output_with_delta = self.output_function(perturbed_input)
            gradient[i] = (output_with_delta - output_data) / delta
        return gradient

    def update_eoa_matrix_advanced(self, gradient_matrix, local_error, error_tolerance):
        U, s, Vt = np.linalg.svd(gradient_matrix)
        if local_error > error_tolerance:
            adjustment_factor = np.sqrt(local_error / error_tolerance)
            s_adjusted = s * adjustment_factor
        else:
            s_adjusted = s
        S_adjusted = np.diag(s_adjusted)
        adjusted_gradient_matrix = U @ S_adjusted @ Vt
        eoa_matrix_updated = np.linalg.pinv(adjusted_gradient_matrix)  
        return eoa_matrix_updated





    def is_within_eoa(self, node, query):
        """Determines whether a given query point falls within the EOA of a specific node in the model. This is critical for deciding whether to use an existing approximation or to create a new one."""
        input_data, output_data, gradient_matrix, eoa_matrix = node.data
        diff_vector = query - input_data
        transformed_diff = np.dot(eoa_matrix, diff_vector)
        if np.dot(transformed_diff, transformed_diff) <= 1:
            return True
        else:
            return False



    def calculate_jacobian(self, input_data):
        """Computes the Jacobian matrix of the output function with respect to the input data using complex step differentiation. This provides a highly accurate method for sensitivity analysis."""
        jacobian = np.zeros((len(input_data), len(input_data)), dtype=np.complex128)
        step_size = 1e-20  # A very small step size
        for i in range(len(input_data)):
            perturbed_input = np.array(input_data, dtype=np.complex128)
            perturbed_input[i] += step_size * 1j  # Apply a complex step
            derivative = self.output_function(perturbed_input)
            jacobian[:, i] = np.imag(derivative) / step_size
        return np.real(jacobian)  # Return the real part, as the imaginary part should be negligible


    def _insert_kd_tree(self, current, new_node, depth):
        if current is None:
            return new_node

        # Cycle through dimensions
        axis = depth % len(new_node.data[0]) 

        # Ensure both points are numpy arrays to avoid shape issues
        current_point = np.array(current.data[0])
        new_point = np.array(new_node.data[0])

        if new_point[axis] < current_point[axis]:
            if current.left is None:
                current.left = new_node
                new_node.axis = axis
            else:
                self._insert_kd_tree(current.left, new_node, depth + 1)
        else:
            if current.right is None:
                current.right = new_node
                new_node.axis = axis
            else:
                self._insert_kd_tree(current.right, new_node, depth + 1)
        return current


    def find_nearest_approximation(self, input_data):
        """
        Finds the nearest node (approximation) to the given input data using the k-d tree.
        """
        if self.root is None:
            return None
        return self._search_nearest(self.root, input_data, 0, None, float('inf'))[0]

    
    def _search_nearest(self, node, target, depth, best=None, best_dist=float('inf')):
        if node is None:
            return best, best_dist

        node_data = node.data[0]
        axis = depth % len(node_data)

        here_dist = np.linalg.norm(node_data - target)
        if here_dist < best_dist:
            best_dist = here_dist
            best = node

        diff = target[axis] - node_data[axis]
        go_left = diff < 0
        good_side = node.left if go_left else node.right
        bad_side = node.right if go_left else node.left

        best, best_dist = self._search_nearest(good_side, target, depth + 1, best, best_dist)

        if diff**2 < best_dist:
            # There might be a closer point on the 'bad' side of the partition.
            best, best_dist = self._search_nearest(bad_side, target, depth + 1, best, best_dist)

        return best, best_dist


    def _distance(self, point1, point2):
        """Calculates the Euclidean distance between two points. This is used during the search process to find the nearest approximation based on input data."""
        return np.linalg.norm(point1 - point2)
    
    def calculate_local_error(self, actual_output, predicted_output):
        """Computes the error between the actual output and the predicted output. This error is used to assess the accuracy of approximations and to decide on updates or expansions of the EOA."""
        return np.linalg.norm(actual_output - predicted_output)
    
    def insert_or_update_approximation(self, input_data, output_data):
        """Either inserts a new approximation or updates an existing one based on the calculated local error relative to the error tolerance. This method streamlines the process of maintaining accurate approximations within the model."""
        nearest = self.find_nearest_approximation(input_data)
        if nearest and self.calculate_local_error(output_data, nearest.data[1]) < self.error_tolerance:
            self.update_approximation(nearest, input_data, output_data)
        else:
            self.add_approximation(input_data, output_data)

    def adapt_approximation(self, input_data, actual_output):
        """Adapts the model by either adding a new approximation or updating an existing one, depending on whether the nearest node's approximation falls within the error tolerance. It ensures the model remains up-to-date and accurate."""
        nearest_node = self.find_nearest_approximation(input_data)
        if nearest_node is None:
            self.add_approximation(input_data, actual_output)
            return
        predicted_output = nearest_node.data[1]
        error = self.calculate_local_error(actual_output, predicted_output)
        if error < self.error_tolerance:
            self.update_approximation(nearest_node, input_data, actual_output)
        else:
            self.insert_or_update_approximation(input_data, actual_output)






