import torch

class Simulator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        pass

    def forward(self, x):
        pass

    @property
    def domain(self):
        pass

    @property
    def range(self):
        pass

    def process_input(self, x):
        pass

    def process_output(self, x):
        pass

    def plot_data(self, x, y, v, Jvp, file_path="plot.png", title="NS Sample Plot"):
        pass
