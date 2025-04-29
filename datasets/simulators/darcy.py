# simulators/darcy.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from groundwater.utils import GaussianRandomField, plot_fields
from groundwater.devito_op import GroundwaterModel
from .base import Simulator  # Assuming base class defines interface
import time # For potential timing/debugging

class DarcySimulator(Simulator):
    def __init__(self, size=256, T=1.0, dtype=torch.float32, fd_epsilon=1e-4): # Add fd_epsilon
        super().__init__()
        self.size = size
        self.dtype = dtype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Important: Ensure GroundwaterModel is compatible with requires_grad=False
        # if you want to *guarantee* no AD is used internally during forward passes
        # for finite differences. Detach inputs if necessary.
        self.model = GroundwaterModel(size)
        self.T = T
        self._fd_epsilon = fd_epsilon # Epsilon for finite differences
        print(f"DarcySimulator initialized on device: {self.device}")


    def sample(self):
        # Step 1: Sample from Gaussian Random Field
        grf = GaussianRandomField(2, self.size, alpha=2, tau=4)
        u_samples = grf.sample(1)
        
        # Ensure sample is on the correct device
        return torch.tensor(u_samples[0], dtype=self.dtype, device=self.device)

    def forward(self, u, enable_grad=False): # Add enable_grad flag
        """
        Runs the forward simulation.

        Args:
            u (torch.Tensor): Input tensor.
            enable_grad (bool): If True, run within torch.enable_grad(), otherwise torch.no_grad().
                                Crucial for compatibility with autograd vs finite differences.
        """
        # Ensure input is on the correct device
        u = u.to(self.device, dtype=self.dtype)

        # Zero forcing term
        f = torch.zeros((self.size, self.size), dtype=self.dtype, device=self.device)

        # Choose context based on enable_grad flag
        context = torch.enable_grad() if enable_grad else torch.no_grad()

        with context:
            # Run the model (GroundwaterModel -> GroundwaterLayer)
            if u.ndim == 3: # Batch dimension assumed by GroundwaterModel? Check its forward.
                # If GroundwaterModel doesn't support batch, iterate here.
                # Assuming it expects (N, size, size) or iterates internally
                # Let's assume model needs (size, size) input based on previous code
                results = []
                for i in range(u.shape[0]):
                     u_i = u[i] # Assuming shape is (N, size, size)
                     # Ensure u_i has correct shape, e.g., u_i.squeeze() if needed
                     out = self.model(u_i, f) # Call model within context
                     results.append(out)
                output = torch.stack(results)

            elif u.ndim == 2: # Single sample (size, size)
                 output = self.model(u, f) # Call model within context
            else:
                 raise ValueError(f"Unsupported input ndim: {u.ndim}. Expected 2 or 3.")

            # If enable_grad=True, output will be attached to the graph.
            # If enable_grad=False, it won't be.
            return output

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the JVP using central finite differences.
        J_x(v) ≈ (forward(x + εv) - forward(x - εv)) / (2ε)
        Calls forward with enable_grad=False.
        """
        x = x.to(self.device, dtype=self.dtype)
        v = v.to(self.device, dtype=self.dtype)

        if x.shape != v.shape:
             raise ValueError(f"Shape mismatch: x {x.shape} vs v {v.shape}")

        epsilon = self._fd_epsilon

        # Ensure forward calls within JVP do not use AD graph tracking
        # No need for extra torch.no_grad() context here, as forward handles it.
        y_plus = self.forward(x + epsilon * v, enable_grad=False)
        y_minus = self.forward(x - epsilon * v, enable_grad=False)

        jvp_result = (y_plus - y_minus) / (2 * epsilon)

        return jvp_result


    def plot_data(self, inputs, outputs, vec=None, jvp_res=None, file_path="darcy_plot.png", title="Darcy Simulator Results"):
        # Ensure data is on CPU for plotting
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.cpu().numpy()
        else:
            inputs_np = inputs
        if isinstance(outputs, torch.Tensor):
            outputs_np = outputs.cpu().numpy()
        else:
            outputs_np = outputs
        if vec is not None and isinstance(vec, torch.Tensor):
             vec_np = vec.cpu().numpy().reshape(inputs_np.shape) # Reshape if needed
        else:
             vec_np = vec # Can be None
        if jvp_res is not None and isinstance(jvp_res, torch.Tensor):
             jvp_res_np = jvp_res.cpu().numpy().reshape(outputs_np.shape) # Reshape if needed
        else:
             jvp_res_np = jvp_res # Can be None

        num_plots = 2 + (1 if vec_np is not None else 0) + (1 if jvp_res_np is not None else 0)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        fig.suptitle(title)

        im = axes[0].imshow(np.exp(inputs_np))
        axes[0].set_title("Input exp(u(x))")
        fig.colorbar(im, ax=axes[0])

        im = axes[1].imshow(outputs_np)
        axes[1].set_title("Output p(x)")
        fig.colorbar(im, ax=axes[1])

        plot_idx = 2
        if vec_np is not None:
            im = axes[plot_idx].imshow(vec_np)
            axes[plot_idx].set_title("Vector v")
            fig.colorbar(im, ax=axes[plot_idx])
            plot_idx += 1
        
        if jvp_res_np is not None:
             im = axes[plot_idx].imshow(jvp_res_np)
             axes[plot_idx].set_title("JVP Result")
             fig.colorbar(im, ax=axes[plot_idx])
             plot_idx += 1


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(file_path)
        plt.close(fig) # Close the figure to free memory


    @property
    def domain(self):
        return self.size * self.size

    @property
    def range(self):
        return self.size * self.size
