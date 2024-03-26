# In-Situ Adaptive Tabulation (ISAT) Implementation

This Python module implements the In-Situ Adaptive Tabulation (ISAT) algorithm, leveraging advanced data structures and numerical methods for efficient function approximation and dynamics system simulation. ISAT is essential for applications requiring rapid evaluations of complex, computationally intensive functions, with this implementation incorporating a KD Tree for efficient input-output mapping storage and retrieval.
This software is experimental and a work in progress, please keep this in mind
-[ISAT webpage](https://apm.byu.edu/prism/index.php/Projects/InSituAdaptiveTabulation)

## Features

- **Efficient Function Approximation**: Uses a KD Tree for quick nearest approximation searches based on input.
- **Adaptive and Dynamic**: Dynamically updates the KD Tree to improve approximations based on customizable error tolerances.
- **Gradient Support**: Includes functionalities for updating the Ellipsoid of Accuracy (EOA) using gradient calculations, facilitating precise control over approximation accuracies.
- **JAX Integration**: Utilizes JAX for gradient calculations and system dynamics simulations, offering potential for GPU acceleration and autodifferentiation capabilities.


## Want to help?
I didn't realize there was an [improved algorithm](https://tcg.mae.cornell.edu/pubs/Lu_LRP_JCP_09.pdf) when developing this software. This improvement can increase performance by adding more table-searching methods, affine space reduction, error correction, ellipsoid of inaccuracy, and improved EOA growing techniques
Also JAX not supporting KD-Trees is a bummer, it may be worth exploring different data structures native to JAX

## References

-[Storage and Retrieval of Optimal Control](https://apm.byu.edu/prism/uploads/Projects/isat_details.pdf) by J. D. Hedengren and T. F. Edgar, from The University of Texas at Austin, discusses the application of the ISAT algorithm in storage and retrieval of optimal control, focusing on its benefits in real-time control scenarios where computational cycle times are critical .

-[Computationally Efficient Implementation of Combustion Chemistry Using In Situ Adaptive Tabulation](https://tcg.mae.cornell.edu/pubs/Pope_CTM_97.pdf) by S.B. Pope, from the Sibley School of Mechanical and Aerospace Engineering, Cornell University, Ithaca NY, introduces and demonstrates the ISAT technique for reducing computational time in detailed chemistry within reactive flow calculations. The method, aimed at addressing the high computational demand in simulating detailed combustion chemistry, showcases a significant speed-up factor, making detailed kinetic mechanisms feasible in calculations of turbulent combustion​​.
