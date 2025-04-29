import os
import numpy as np
import torch

from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import loadmat


###############################################################
# This code is adapted from
# https://github.com/shawnrosofsky/PINO_Applications
#
###############################################################

class SWE_Nonlinear(torch.nn.Module):
    def __init__(self,
                 xmin=0.0,
                 xmax=1.0,
                 ymin=0.0,
                 ymax=1.0,
                 s1=128,
                 s2=128,
                 g=1.0,
                 nu=0.002,
                 T=1.0,
                 dt=1.0e-3,
                 adaptive=False,
                 nburn=10,
                 nsteps=100):

        super().__init__()

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.s1 = s1
        self.s2 = s2
        self.g = g
        self.nu = nu
        self.dt = dt
        self.T = T
        self.adaptive = adaptive
        self.nburn = nburn
        self.nsteps = nsteps

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

        self.x = torch.linspace(xmin, xmax, s1 + 1, device=self.device, dtype=self.dtype)[:-1]
        self.y = torch.linspace(ymin, ymax, s2 + 1, device=self.device, dtype=self.dtype)[:-1]
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')

    def Dx(self, data):
        return (torch.roll(data, -1, dims=0) - torch.roll(data, 1, dims=0)) / (2 * self.dx)

    def Dy(self, data):
        return (torch.roll(data, -1, dims=1) - torch.roll(data, 1, dims=1)) / (2 * self.dy)

    def Dxx(self, data):
        return (torch.roll(data, -1, dims=0) - 2 * data + torch.roll(data, 1, dims=0)) / self.dx ** 2

    def Dyy(self, data):
        return (torch.roll(data, -1, dims=1) - 2 * data + torch.roll(data, 1, dims=1)) / self.dy ** 2

    def calc_RHS(self, h, u, v):
        h_rhs = -self.Dx(h * u) - self.Dy(h * v)
        u_rhs = -self.Dx(h * u ** 2 + 0.5 * self.g * h ** 2) - self.Dy(h * u * v) + self.nu * (self.Dxx(u) + self.Dyy(u))
        v_rhs = -self.Dx(h * u * v) - self.Dy(h * v ** 2 + 0.5 * self.g * h ** 2) + self.nu * (self.Dxx(v) + self.Dyy(v))
        return h_rhs, u_rhs, v_rhs

    def rk4_step(self, h, u, v):
        def step(h, u, v, dt_frac, h_rhs, u_rhs, v_rhs):
            return h + dt_frac * self.dt * h_rhs, u + dt_frac * self.dt * u_rhs, v + dt_frac * self.dt * v_rhs

        k1 = self.calc_RHS(h, u, v)
        h1, u1, v1 = step(h, u, v, 0.5, *k1)

        k2 = self.calc_RHS(h1, u1, v1)
        h2, u2, v2 = step(h, u, v, 0.5, *k2)

        k3 = self.calc_RHS(h2, u2, v2)
        h3, u3, v3 = step(h, u, v, 1.0, *k3)

        k4 = self.calc_RHS(h3, u3, v3)

        h_new = h + (self.dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        u_new = u + (self.dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        v_new = v + (self.dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

        return h_new, u_new, v_new

    def advance(self, h, u, v):
        t = 0.0
        while t < self.T:
            h, u, v = self.rk4_step(h, u, v)
            t += self.dt
        return h, u, v

    def sample(self):
        h0 = 1.0 + 0.1 * torch.exp(-((self.X - 0.5) ** 2 + (self.Y - 0.5) ** 2) / (2 * 0.05 ** 2))
        u0 = torch.zeros_like(h0)
        v0 = torch.zeros_like(h0)

        h = h0
        u = u0
        v = v0

        for _ in range(self.nburn):
            h, u, v = self.advance(h, u, v)

        return h, u, v

    # def forward(self, h):
    #     u = torch.zeros_like(h)
    #     v = torch.zeros_like(h)
    #     for i in range(self.nsteps):
    #         if (i + 1) % 10 == 0:
    #             print(f"SWE Simulator Step {i + 1} of {self.nsteps}")
    #         h, u, v = self.advance(h, u, v)
    #     return h
    def forward(self, x):
        if isinstance(x, tuple):
            h, u, v = x
        else:
            h = x
            u = torch.zeros_like(h)
            v = torch.zeros_like(h)

        for i in range(self.nsteps):
            if (i + 1) % 10 == 0:
                print(f"SWE Simulator Step {i + 1} of {self.nsteps}")
            h, u, v = self.advance(h, u, v)

        return h  # or return h, u, v if you need them all


    def plot_data(self, x, y, v, Jvp, file_path="plot.png", title="SWE Sample Plot"):
        def prepare_field(field):
            return {'height': field.cpu().numpy()}

        x_data = prepare_field(x)
        y_data = prepare_field(y)
        v_data = prepare_field(v)
        jvp_data = prepare_field(Jvp)

        fig, axs = plt.subplots(4, 1, figsize=(5, 20))
        fig.suptitle(title)

        plot_data = [
            (0, x_data, ['Input height']),
            (1, y_data, ['Output height']),
            (2, v_data, ['Eigenvector height']),
            (3, jvp_data, ['Jvp height'])
        ]

        for row, data, titles in plot_data:
            axs[row].imshow(data['height'], cmap='jet')
            axs[row].set_title(titles[0])

        plt.savefig(file_path)
        plt.close()

    @property
    def domain(self):
        return self.s1 * self.s2

    @property
    def range(self):
        return self.s1 * self.s2

if __name__ == "__main__":

    simulator = SWE_Nonlinear(
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        s1=128, s2=128,
        g=1.0, nu=0.001,
        T=1.0, dt=1e-3,
        nburn=10, nsteps=100
    )

    h0, u0, v0 = simulator.sample()  # this gives a warm-started state
    hT = simulator((h0, u0, v0))

    simulator.plot_data(
    x=h0,
    y=hT,
    v=torch.zeros_like(h0),       # eigenvector placeholder
    Jvp=torch.zeros_like(h0),     # JvP placeholder
    file_path="swe_plot.png",
    title="SWE Simulation Output"
    )




# class SWE_Nonlinear(torch.nn.Module):
#     def __init__(self,
#                  xmin=0,
#                  xmax=1,
#                  ymin=0,
#                  ymax=1,
#                  Nx=100,
#                  Ny=100,
#                  g=1.0,
#                  nu=0.001,
#                  dt=1.0e-3,
#                  tend=0.5,
#                  device=None,
#                  dtype=torch.float64,                                                  
#                  ):
        
#         super().__init__()

#         self.xmin = xmin
#         self.xmax = xmax
#         self.ymin = ymin
#         self.ymax = ymax
#         self.Nx = Nx
#         self.Ny = Ny
#         self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
#         self.dtype = torch.float64
#         x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
#         y = torch.linspace(ymin, ymax, Ny + 1, device=device, dtype=dtype)[:-1]
#         self.x = x
#         self.y = y
#         self.dx = x[1] - x[0]
#         self.dy = y[1] - y[0]
#         self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
#         self.g = g
#         self.nu = nu
#         self.h = torch.zeros_like(self.X, device=device)
#         self.h0 = torch.zeros_like(self.h, device=device)
#         self.u = torch.zeros_like(self.X, device=device)
#         self.u0 = torch.zeros_like(self.u, device=device)
#         self.v = torch.zeros_like(self.X, device=device)
#         self.v0 = torch.zeros_like(self.v, device=device)
#         self.dt = dt
#         self.tend = tend
#         self.t = 0
#         self.it = 0
#         self.H = []
#         self.U = []
#         self.V = []
#         self.T = []
    
#     def initialize_gaussian(self,amp=0.1, sigma=0.05, loc=[0.5,0.5]):
#         loc_x = loc[0]
#         loc_y = loc[1]

#         # There are three conserved quantities - initialize
#         h0 = 1.0 + amp*torch.exp(-((self.X-loc_x)**2/(2*(sigma)**2) + (self.Y-loc_y)**2/(2*(sigma)**2)))
#         return h0
        
        
#     # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
#     def CD_i(self, data, axis, dx):
#         data_m2 = torch.roll(data,shifts=2,dims=axis)
#         data_m1 = torch.roll(data,shifts=1,dims=axis)
#         data_p1 = torch.roll(data,shifts=-1,dims=axis)
#         data_p2 = torch.roll(data,shifts=-2,dims=axis)
#         data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
#         return data_diff_i

#     def CD_ij(self, data, axis_i, axis_j, dx, dy):
#         data_diff_i = self.CD_i(data,axis_i,dx)
#         data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
#         return data_diff_ij

#     def CD_ii(self, data, axis, dx):
#         data_m2 = torch.roll(data,shifts=2,dims=axis)
#         data_m1 = torch.roll(data,shifts=1,dims=axis)
#         data_p1 = torch.roll(data,shifts=-1,dims=axis)
#         data_p2 = torch.roll(data,shifts=-2,dims=axis)
#         data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
#         return data_diff_ii

#     def Dx(self, data):
#         data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
#         return data_dx
    
#     def Dy(self, data):
#         data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
#         return data_dy

#     def Dxx(self, data):
#         data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
#         return data_dxx

#     def Dyy(self, data):
#         data_dyy = self.CD_ii(data, axis=1, dx=self.dy)
#         return data_dyy

#     def calc_RHS(self, h, u, v):
#         h_flux_x = self.Dx(h*u)
#         h_flux_y = self.Dy(h*v)
#         u_flux_x = self.Dx(h*u**2 + 0.5*self.g*h**2)
#         u_flux_y = self.Dy(h*u*v)
#         u_xx = self.Dxx(u)
#         u_yy = self.Dyy(u)
#         u_visc = self.nu*(u_xx + u_yy)
#         v_flux_x = self.Dx(h*u*v)
#         v_flux_y = self.Dy(h*v**2 + 0.5*self.g*h**2)
#         v_xx = self.Dxx(v)
#         v_yy = self.Dyy(v)
#         v_visc = self.nu*(v_xx + v_yy)
        
#         h_RHS = -(h_flux_x + h_flux_y)
#         u_RHS = -(u_flux_x + u_flux_y) + u_visc
#         v_RHS = -(v_flux_x + v_flux_y) + v_visc
#         return h_RHS, u_RHS, v_RHS
        
#     def update_field(self, field, RHS, step_frac):
#         field_new = field + self.dt*step_frac*RHS
#         return field_new
        

#     def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
#         field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
#         return field_new

#     def rk4(self, h, u, v, t=0):
#         h_RHS1, u_RHS1, v_RHS1 = self.calc_RHS(h, u, v)
#         # display(h_RHS1)
#         t1 = t + 0.5*self.dt
#         h1 = self.update_field(h, h_RHS1, step_frac=0.5)
#         u1 = self.update_field(u, u_RHS1, step_frac=0.5)
#         v1 = self.update_field(v, v_RHS1, step_frac=0.5)
        
#         h_RHS2, u_RHS2, v_RHS2 = self.calc_RHS(h1, u1, v1)
#         # display(h_RHS2)

#         t2 = t + 0.5*self.dt
#         h2 = self.update_field(h, h_RHS2, step_frac=0.5)
#         u2 = self.update_field(u, u_RHS2, step_frac=0.5)
#         v2 = self.update_field(v, v_RHS2, step_frac=0.5)
        
#         h_RHS3, u_RHS3, v_RHS3 = self.calc_RHS(h2, u2, v2)
#         # display(h_RHS3)
#         t3 = t + self.dt
#         h3 = self.update_field(h, h_RHS3, step_frac=1.0)
#         u3 = self.update_field(u, u_RHS3, step_frac=1.0)
#         v3 = self.update_field(v, v_RHS3, step_frac=1.0)
        
#         h_RHS4, u_RHS4, v_RHS4 = self.calc_RHS(h3, u3, v3)
#         # display(h_RHS4)
        
#         t_new = t + self.dt
#         h_new = self.rk4_merge_RHS(h, h_RHS1, h_RHS2, h_RHS3, h_RHS4)
#         u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
#         v_new = self.rk4_merge_RHS(v, v_RHS1, v_RHS2, v_RHS3, v_RHS4)
        
#         return h_new, u_new, v_new, t_new
    
#     def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
#         plt.ion()
#         fig = plt.figure(fig_num)
#         plt.cla()
#         plt.clf()
        
#         c = plt.pcolormesh(self.X, self.Y, self.h, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
#         fig.colorbar(c)
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.axis('equal')
#         plt.axis('square')
#         plt.draw() 
#         plt.pause(1e-17)
#         plt.show()
        
        
#     def driver(self, h0, save_interval=10, plot_interval=0):
#         # plot results
#         # t,it = get_time(time)
# #         display(u0[:self.Nx,:self.Ny].shape)
#         # self.u0 = u0[:self.Nx]
#         self.h0 = h0[:self.Nx,:self.Ny]
#         # self.u0 = u0[:self.Nx,:self.Ny]
#         # self.v0 = v0[:self.Nx,:self.Ny]
#         self.h = self.h0
#         self.u = self.u0
#         self.v = self.v0
#         self.t = 0
#         self.it = 0
#         self.T = []
#         self.H = []
#         self.U = []
#         self.V = []
        
#         if plot_interval != 0 and self.it % plot_interval == 0:
#             self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
#         if save_interval != 0 and self.it % save_interval == 0:
#             self.H.append(self.h)
#             self.U.append(self.u)
#             self.V.append(self.v)
#             self.T.append(self.t)
#         # Compute equations
#         while self.t < self.tend:
# #             print(f"t:\t{self.t}")
#             self.h, self.u, self.v, self.t = self.rk4(self.h, self.u, self.v, self.t)
            
#             self.it += 1
#             if plot_interval != 0 and self.it % plot_interval == 0:
#                 self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
#             if save_interval != 0 and self.it % save_interval == 0:
#                 self.H.append(self.h)
#                 self.U.append(self.u)
#                 self.V.append(self.v)
#                 self.T.append(self.t)

#         return torch.stack(self.H), torch.stack(self.U), torch.stack(self.V)