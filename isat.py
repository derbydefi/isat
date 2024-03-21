import numpy as np



class BSTNode:
    def __init__(self, left=None, right=None, data=None):
        """Initializes a binary search tree node with optional left and right children and data. The data typically includes input values, output values, and information related to the Ellipsoid of Accuracy (EOA)."""
        self.left = left
        self.right = right
        self.data = data  

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

    def initialize_bst(self):
        """Ensures the binary search tree (BST) has a root node. This method is used to prepare the BST for data insertion when the model is first used."""
        if self.root is None:
            self.root = BSTNode()



    def add_approximation(self, input_data, output_data):
        """Adds a new approximation to the ISAT model by creating a BST node with given input and output data. If the tree is empty, the new node becomes the root; otherwise, it is inserted according to BST rules."""

        # Calculate the gradient matrix for the new data point
        gradient_matrix = self.recalculate_gradient(input_data, output_data)
        
        # Initialize the EOA matrix - this example assumes a simple initial EOA, but it should be adapted based on your requirements
        eoa_matrix = np.eye(len(input_data))  # Identity matrix as a starting point
        
        # Add the new node with input data, output data, gradient matrix, and EOA matrix
        new_node = BSTNode(data=(input_data, output_data, gradient_matrix, eoa_matrix))
        
        if self.root is None:
            self.root = new_node
        else:
            self._insert_bst(self.root, new_node)
    def query_approximation(self, input_data):
        """
        Queries the ISAT model for an approximation of the given input data.
        If an existing approximation within the error tolerance is found, it is used.
        Otherwise, a new approximation is computed, added to the model, and returned.
        """
        # Attempt to find the nearest approximation in the model
        nearest_node = self.find_nearest_approximation(input_data)
        if nearest_node is not None:
            # Calculate local error to decide if the approximation is acceptable
            predicted_output = nearest_node.data[1]  # Assuming this is the output value
            actual_output = self.output_function(input_data)
            error = self.calculate_local_error(actual_output, predicted_output)
            
            if error <= self.error_tolerance:
                # If the error is within tolerance, use the predicted output
                return predicted_output
            else:
                # If the error is too large, update the approximation
                self.update_approximation(nearest_node, input_data, actual_output)
                return actual_output
        else:
            # If no approximation was found, compute and add a new one
            actual_output = self.output_function(input_data)
            self.add_approximation(input_data, actual_output)
            return actual_output



    def update_approximation(self, node, new_input, new_output):
        input_data, output_data, gradient_matrix, eoa_matrix = node.data
        A_prime = self.calculate_jacobian(new_input)
        local_error = self.calculate_local_error(output_data, new_output)
        
        eoa_matrix_updated = self.update_eoa_matrix_advanced(gradient_matrix, local_error, self.error_tolerance)
        
        node.data = (input_data, new_output, A_prime, eoa_matrix_updated)



    def recalculate_gradient(self, input_data, output_data):
        """Calculates the gradient of the output with respect to the input by applying a small perturbation to the input. This method is used for sensitivity analysis and is crucial for updating the EOA accurately."""
        delta = 1e-5  
        gradient = np.zeros_like(input_data)
        for i in range(len(input_data)):
            input_plus_delta = np.copy(input_data)
            input_plus_delta[i] += delta
            output_plus_delta = self.output_function(input_plus_delta)  
            gradient[i] = (output_plus_delta - output_data) / delta
        return gradient

    def update_eoa_matrix_advanced(self, gradient_matrix, local_error, error_tolerance):
        # No need to unpack node.data as we're directly using the provided arguments
        # Use SVD for the updated gradient matrix
        U, s, Vt = np.linalg.svd(gradient_matrix)
        
        # Adjust the EOA based on local error compared to error tolerance
        if local_error > error_tolerance:
            adjustment_factor = np.sqrt(local_error / error_tolerance)
            s_adjusted = s * adjustment_factor
        else:
            s_adjusted = s

        S_adjusted = np.diag(s_adjusted)
        adjusted_gradient_matrix = U @ S_adjusted @ Vt

        # Inverse of the adjusted matrix to update the EOA shape
        pseudo_inverse_adjusted = np.linalg.pinv(adjusted_gradient_matrix)
        eoa_matrix_updated = np.linalg.sqrtm(pseudo_inverse_adjusted)
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


    def _insert_bst(self, current, new_node):
        """Inserts a new node into the binary search tree in the correct position according to BST rules. This method ensures the tree remains ordered, facilitating efficient search and retrieval."""
        if new_node.data[0] < current.data[0]:
            if current.left is None:
                current.left = new_node
            else:
                self._insert_bst(current.left, new_node)
        else:
            if current.right is None:
                current.right = new_node
            else:
                self._insert_bst(current.right, new_node)

    def find_nearest_approximation(self, input_data):
        """Searches the BST for the node that best approximates a given input data point. This method is key for retrieving approximations and deciding on further actions like update or addition."""
        if self.root is None:
            return None
        return self._search_bst(self.root, input_data)
    
    def _search_bst(self, current, target, closest=None):
        """Performs a recursive search through the BST to find the nearest approximation to a given target. It prioritizes nodes within the EOA but defaults to the closest node if no suitable match is found within the EOA."""
        if current is None:
            return closest
        if self.is_within_eoa(current, target):
            # If within EOA, this node is a strong candidate
            return current
        # Even if not within EOA, update closest if this node is closer than the current closest
        if closest is None or self._distance(target, current.data[0]) < self._distance(target, closest.data[0]):
            closest = current
        # Continue searching in the direction that target dictates
        if target < current.data[0]:
            return self._search_bst(current.left, target, closest)
        else:
            return self._search_bst(current.right, target, closest)


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






