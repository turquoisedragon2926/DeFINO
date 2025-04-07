import torch
from .base import ReducedModel
import matplotlib.pyplot as plt
import threading

class FIMReducedModel(ReducedModel):
    def __init__(self, eigen_value_fraction, eigen_vector_count):
        self.eigen_value_fraction = eigen_value_fraction
        self.eigen_vector_count = eigen_vector_count
        self.thread_lock = threading.Lock()

    def get_direction(self, simulator, x):
        with self.thread_lock:
            eigen_count = self.eigen_count(simulator)

            y, jvp_func = torch.func.jvp(simulator, x)
            eigenvectors = torch.randn((simulator.domain, eigen_count))
            
            Z = torch.randn((simulator.range, eigen_count)).to(x.device)
            B, R = torch.linalg.qr(Z)
            
            Q = torch.zeros((simulator.domain, eigen_count))
            for j in range(eigen_count):
                print(f"Computing FIM Eigenvector {j + 1} of {eigen_count}")

                probe_vector = B[:, j].reshape(y.shape)
                Q[:, j] = jvp_func(probe_vector)[0].reshape((simulator.domain,))

            U, S, V = torch.linalg.svd(Q)
            eigenvectors = U[:, :eigen_count]
            
            return eigenvectors, S
    
    def plot_decay(self, s, path, title):
        plt.plot(s)
        plt.title(title)
        plt.xlabel("Eigenvector")
        plt.ylabel("Singular Value")
        plt.savefig(path)
        plt.close()

    def eigen_count(self, simulator):
        if self.eigen_vector_count is not None:
            return self.eigen_vector_count
        return int(simulator.domain * self.eigen_value_fraction)

