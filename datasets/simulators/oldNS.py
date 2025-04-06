import torch
import torch.autograd.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .base import Simulator

'''
input: velocity fields vx, vy
output: velocity fields vx, vy
'''

class OldNavierStokesSimulator(Simulator):
    def __init__(self, N, L, dt, nu, nburn=200, nsteps=500):
        super().__init__()
        self.N = N
        self.L = L
        self.dt = dt
        self.nu = nu
        self.nburn = nburn
        self.nsteps = nsteps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_fourier_space()

    def setup_fourier_space(self):
        klin = 2.0 * torch.pi / self.L * torch.arange(-self.N//2, self.N//2, device=self.device)
        self.kx, self.ky = torch.meshgrid(klin, klin, indexing='ij')
        self.kx = torch.fft.ifftshift(self.kx)
        self.ky = torch.fft.ifftshift(self.ky)
        self.kSq = self.kx**2 + self.ky**2
        self.kSq_inv = 1.0 / self.kSq
        self.kSq_inv[self.kSq == 0] = 1

        xlin = torch.linspace(0, self.L, self.N, device=self.device)
        self.xx, self.yy = torch.meshgrid(xlin, xlin, indexing='ij')

        kmax = torch.max(klin)
        self.dealias = ((torch.abs(self.kx) < (2./3.)*kmax) & (torch.abs(self.ky) < (2./3.)*kmax)).float()
        
    def sample(self):
            
        freq_x = torch.normal(mean=8.0, std=0.3, size=(1,), device=self.device).item()
        freq_y = torch.normal(mean=16.0, std=0.5, size=(1,), device=self.device).item()
        phase_x = torch.normal(mean=0.0, std=1., size=(1,), device=self.device).item()
        phase_y = torch.normal(mean=0.0, std=1., size=(1,), device=self.device).item()
    
        vx = -torch.sin(freq_y * torch.pi * self.yy + phase_y)
        vy = torch.sin(freq_x * torch.pi * self.xx + phase_x)
        
        for i in range(self.nburn):
            vx, vy = self.forward_step(vx, vy)
        
        return torch.stack([vx, vy], dim=0)
        
    def forward_step(self, vx, vy):
        dvx_x, dvx_y = self.grad(vx)
        dvy_x, dvy_y = self.grad(vy)
            
        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)
        
        rhs_x = self.apply_dealias(rhs_x)
        rhs_y = self.apply_dealias(rhs_y)

        vx = vx + self.dt * rhs_x
        vy = vy + self.dt * rhs_y
        
        div_rhs = self.div(rhs_x, rhs_y)
        P = self.poisson_solve(div_rhs)
        dPx, dPy = self.grad(P)
        
        vx = vx - self.dt * dPx
        vy = vy - self.dt * dPy
        
        vx = self.diffusion_solve(vx)
        vy = self.diffusion_solve(vy)
        
        return vx, vy
        
    def forward(self, x):
        vx = x[0]
        vy = x[1]
        
        for i in range(self.nsteps):
            if i % 100 == 0:
                print(f"OLD NS Simulator Step {i + 1} of {self.nsteps}")
            vx, vy = self.forward_step(vx, vy)
        
        return torch.stack([vx, vy], dim=0)

    def grad(self, v):
        v_hat = torch.fft.fftn(v)
        return (
            torch.real(torch.fft.ifftn(1j * self.kx * v_hat)),
            torch.real(torch.fft.ifftn(1j * self.ky * v_hat))
        )

    def div(self, vx, vy):
        return (
            torch.real(torch.fft.ifftn(1j * self.kx * torch.fft.fftn(vx))) +
            torch.real(torch.fft.ifftn(1j * self.ky * torch.fft.fftn(vy)))
        )

    def curl(self, vx, vy):
        return (
            torch.real(torch.fft.ifftn(1j * self.ky * torch.fft.fftn(vx))) -
            torch.real(torch.fft.ifftn(1j * self.kx * torch.fft.fftn(vy)))
        )

    def poisson_solve(self, rho):
        rho_hat = torch.fft.fftn(rho)
        phi_hat = -rho_hat * self.kSq_inv
        return torch.real(torch.fft.ifftn(phi_hat))

    def diffusion_solve(self, v):
        v_hat = torch.fft.fftn(v)
        v_hat = v_hat / (1.0 + self.dt * self.nu * self.kSq)
        return torch.real(torch.fft.ifftn(v_hat))

    def apply_dealias(self, f):
        f_hat = self.dealias * torch.fft.fftn(f)
        return torch.real(torch.fft.ifftn(f_hat))

    @property
    def domain(self):
        return 2 * self.N ** 2
    
    @property
    def range(self):
        return 2 * self.N ** 2

    def plot_data(self, x, y, v, Jvp, file_path="plot.png", title="NS Sample Plot"):
        """
        Plot velocity fields and their curls for visualization.
        
        Args:
            x: Input velocity fields
            y: Output velocity fields
            v: Eigenvector velocity fields
            Jvp: Jacobian-vector product velocity fields
            file_path: Path to save the plot
        """
        # Helper function to prepare data
        def prepare_field(field):
            field = field.reshape(2, 128, 128)
            curl = self.curl(field[0], field[1])
            # Convert to numpy arrays for plotting
            return {
                'vx': field[0].cpu().numpy(),
                'vy': field[1].cpu().numpy(),
                'curl': curl.cpu().numpy()
            }
        
        # Prepare all fields
        x_data = prepare_field(x)
        y_data = prepare_field(y)
        v_data = prepare_field(v)
        jvp_data = prepare_field(Jvp)
        
        # Create figure and subplots
        fig, axs = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle(title)
        
        # Data to plot with corresponding titles
        plot_data = [
            (0, x_data, ['Input vx', 'Input vy', 'Input curl']),
            (1, y_data, ['Output vx', 'Output vy', 'Output curl']),
            (2, v_data, ['Eigenvector 1 vx', 'Eigenvector 1 vy', 'Eigenvector 1 curl']),
            (3, jvp_data, ['Jvp 1 vx', 'Jvp 1 vy', 'Jvp 1 curl'])
        ]
        
        # Plot all data
        for row, data, titles in plot_data:
            axs[row, 0].imshow(data['vx'], cmap='RdBu')
            axs[row, 0].set_title(titles[0])
            
            axs[row, 1].imshow(data['vy'], cmap='RdBu')
            axs[row, 1].set_title(titles[1])
            
            axs[row, 2].imshow(data['curl'], cmap='RdBu')
            axs[row, 2].set_title(titles[2])
        
        plt.savefig(file_path)
        plt.close()
