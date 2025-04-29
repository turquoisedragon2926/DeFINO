# datasets/reduced_orders/fim.py
import torch
from .base import ReducedModel
import matplotlib.pyplot as plt
import threading # Keep thread lock if needed
import numpy as np

class FIMReducedModel(ReducedModel):
    def __init__(self, eigen_value_fraction, eigen_vector_count):
        self.eigen_value_fraction = eigen_value_fraction
        self.eigen_vector_count = eigen_vector_count
        # Keep lock if external factors necessitate it, otherwise might not be needed
        # if generate_dataset's ThreadPoolExecutor handles instance isolation.
        # Let's keep it for safety for now.
        self.thread_lock = threading.Lock()
        print(f"FIMReducedModel initialized: eigen_count={eigen_vector_count}")

    # <<< MODIFIED get_direction method >>>
    def get_direction(self, simulator, x):
        """
        Computes directions based on SVD of J^T B using torch.autograd.grad for VJPs.
        J = Jacobian of simulator, B = Orthogonal basis of random output subspace.
        """
        with self.thread_lock:
            device = x.device
            dtype = x.dtype
            eigen_count = self.eigen_count(simulator)
            if eigen_count <= 0:
                 raise ValueError(f"eigen_count must be positive, got {eigen_count}")

            input_dim = simulator.domain
            output_dim = simulator.range

            # --- Ensure input requires grad ---
            x_clone = x.clone().detach().requires_grad_(True)

            # --- Perform forward pass WITH gradient tracking ---
            # This now relies on simulator.forward having enable_grad=True capability
            print(f"FIM: Running forward pass with grad enabled (Input: {x_clone.shape})...")
            y = simulator(x_clone, enable_grad=True)
            original_y_shape = y.shape

            # --- Handle potential non-tensor output ---
            if not isinstance(y, torch.Tensor):
                 y = y[0] # Assume first element is the relevant output
                 print("Warning: Simulator output appears to be a tuple. Using the first element for FIM.")
            
            if y.numel() != output_dim:
                 print(f"Warning: Simulator output size {y.numel()} doesn't match range {output_dim}. Reshaping.")
                 y = y.reshape(output_dim) # Ensure y matches expected output dimension


            # --- Generate orthogonal basis B for output subspace ---
            # Ensure eigen_count doesn't exceed output dimension for QR
            k = min(eigen_count, output_dim)
            if k != eigen_count:
                 print(f"Warning: eigen_count ({eigen_count}) > output_dim ({output_dim}). Using k={k} for basis B.")
                 
            print(f"FIM: Generating {k} orthogonal basis vectors for output space ({output_dim})...")
            Z = torch.randn((output_dim, k), dtype=dtype, device=device)
            B, R = torch.linalg.qr(Z) # B has shape [output_dim, k]

            # --- Compute Q = J^T B using autograd.grad ---
            Q = torch.zeros((input_dim, k), dtype=dtype, device=device) # Q shape [input_dim, k]
            
            print(f"FIM: Computing {k} VJPs (J^T B[:,j])...")
            # Ensure y is contiguous for grad calculation efficiency/correctness
            y_for_grad = y.contiguous()

            for j in range(k):
                # print(f"Computing FIM VJP {j + 1}/{k}") # Verbose
                
                # probe_vector is B[:, j] - shape [output_dim]
                probe_vector = B[:, j]
                probe_vector_reshaped = probe_vector.reshape(original_y_shape)
                
                # Reshape probe_vector to match the original shape of y if needed by grad_outputs
                # If y was originally e.g., (H, W), reshape probe_vector to (H, W)
                # Assuming y is already flattened to [output_dim] based on earlier check.
                # probe_vector_reshaped = probe_vector.reshape(y.shape) # Use if y wasn't flattened

                # --- VJP Calculation using autograd.grad ---
                vjp = torch.autograd.grad(
                    outputs=y_for_grad,     # Output tensor(s) from forward pass
                    inputs=x_clone,       # Input tensor(s) requiring grad
                    grad_outputs=probe_vector_reshaped, # Vector v for J^T v
                    retain_graph=True,    # VERY IMPORTANT: Keep graph for the next VJP in the loop
                    allow_unused=False    # Error if x_clone isn't ancestor of y_for_grad
                )[0] # Result is a tuple of grads w.r.t. inputs; get the first/only one

                if vjp is None:
                     raise RuntimeError(f"torch.autograd.grad returned None for VJP probe {j+1}. Check graph.")

                # Store the resulting VJP (J^T B[:,j]) as a column in Q
                Q[:, j] = vjp.reshape(input_dim,) # Flatten VJP to [input_dim]

            # --- SVD of Q = J^T B ---
            # Clear graph manually if needed, or let context manager handle it
            # y_for_grad = None # Optional: release reference
            # x_clone.grad = None # Optional: clear gradient
            
            print(f"FIM: Computing SVD of Q ({Q.shape})...")
            try:
                # Q has shape [input_dim, k]
                U, S, Vh = torch.linalg.svd(Q, full_matrices=False) # Use economy SVD
                # U has shape [input_dim, k] (Left singular vectors = directions in input space)
                # S has shape [k] (Singular values)
                # Vh has shape [k, k] (Right singular vectors)
                
            except torch.linalg.LinAlgError as e:
                 print(f"Warning: SVD failed ({e}). Returning random orthogonal directions.")
                 random_matrix = torch.randn((input_dim, k), dtype=dtype, device=device)
                 U, _ = torch.linalg.qr(random_matrix)
                 S = torch.zeros(k, device=device, dtype=dtype)

            # --- Select results ---
            # We need top 'eigen_count' directions, but we only computed 'k'
            # If k < eigen_count, we can only return k directions.
            num_results = min(k, eigen_count)
            
            selected_vectors = U[:, :num_results] # Shape [input_dim, num_results]
            selected_values = S[:num_results]   # Shape [num_results]

            print(f"FIM: Selected {selected_vectors.shape[1]} directions based on SVD.")
            # Detach results from computation graph
            return selected_vectors.detach(), selected_values.detach()


    # Keep plot_decay as is (or use the slightly improved version from previous response)
    def plot_decay(self, s, path, title):
         if isinstance(s, torch.Tensor):
             s_np = s.cpu().numpy()
         else:
             s_np = s

         plt.figure(figsize=(8, 5))
         # Use non-negative part for log plot, add small epsilon for zero values
         s_plot = np.maximum(s_np, 1e-30) # Avoid log(0)
         plt.semilogy(s_plot, 'o-')
         plt.title(title)
         plt.xlabel("Index")
         plt.ylabel("Singular Value (log scale)")
         plt.grid(True, which='both', linestyle='--')
         plt.tight_layout()
         plt.savefig(path)
         plt.close()

    # Keep eigen_count as is
    def eigen_count(self, simulator):
        if self.eigen_vector_count is not None:
            # Ensure count is not larger than input dimension
            count = min(self.eigen_vector_count, simulator.domain)
            if count <= 0:
                 print(f"Warning: Calculated eigen_count is {count}. Setting to 1.")
                 return 1
            return count
            
        # Calculate based on fraction, ensure it's at least 1 and integer
        count = int(simulator.domain * self.eigen_value_fraction)
        count = max(1, count) # Ensure at least 1
        # Ensure count is not larger than input dimension
        count = min(count, simulator.domain)
        return count