# In-Situ Adaptive Tabulation (ISAT) Implementation

This Python module implements the In-Situ Adaptive Tabulation (ISAT) algorithm for efficient function approximation. ISAT is particularly useful for applications requiring rapid evaluations of complex, computationally intensive functions. The core of this implementation involves a K-D Tree for storing and retrieving approximations based on input-output mappings. This software is a work in progress and any help would be appreciated!

## Features

- **Efficient Approximation**: Leverages BST to quickly find the nearest approximation to a given input.
- **Adaptive**: Dynamically updates and grows the tree based on error tolerances to improve approximations.
- **Error Tolerance**: Customizable error threshold for deciding when to update or expand approximations.
- **Gradient Calculation**: Supports updating the Ellipsoid of Accuracy (EOA) through gradient recalculations.

## Want to help?
I didn't realize there was an [improved algorithm](https://tcg.mae.cornell.edu/pubs/Lu_LRP_JCP_09.pdf) when developing this software. This improvement can increase performance by adding more table-searching methods, affine space reduction, error correction, ellipsoid of inaccuracy, and improved EOA growing techniques

## References

-[Storage and Retrieval of Optimal Control](https://apm.byu.edu/prism/uploads/Projects/isat_details.pdf) by J. D. Hedengren and T. F. Edgar, from The University of Texas at Austin, discusses the application of the ISAT algorithm in storage and retrieval of optimal control, focusing on its benefits in real-time control scenarios where computational cycle times are critical .

-[Computationally Efficient Implementation of Combustion Chemistry Using In Situ Adaptive Tabulation](https://tcg.mae.cornell.edu/pubs/Pope_CTM_97.pdf) by S.B. Pope, from the Sibley School of Mechanical and Aerospace Engineering, Cornell University, Ithaca NY, introduces and demonstrates the ISAT technique for reducing computational time in detailed chemistry within reactive flow calculations. The method, aimed at addressing the high computational demand in simulating detailed combustion chemistry, showcases a significant speed-up factor, making detailed kinetic mechanisms feasible in calculations of turbulent combustion​​.
