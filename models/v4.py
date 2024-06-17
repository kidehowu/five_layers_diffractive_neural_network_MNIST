import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AngularPropagation(nn.Module):
    def __init__(self, distance, n_padd, pixel_pitch, lambd, size):
        super(AngularPropagation, self).__init__()
        self.size = size  # 200 * 200 neurons in one layer
        self.com_pitch = 2 * n_padd + self.size
        self.dx = pixel_pitch
        self.distance_non_dimension = distance / self.dx  # distance bewteen two layers (3cm)
        self.lambda_non_dimension = lambd / self.dx  # wave length
        self.k_non_dimension = 2 * np.pi / self.lambda_non_dimension  # wave number
        # self.phi (200, 200)
        fx = torch.linspace(-self.com_pitch / 2 + 1, self.com_pitch / 2, self.com_pitch)
        fy = fx
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')

        deter_matrix = 1 - (self.lambda_non_dimension ** 2 / self.com_pitch ** 2) * (FX ** 2 + FY ** 2)
        mask = deter_matrix < 0
        deter_matrix = torch.clamp(deter_matrix, min=0)
        deter_matrix = torch.exp(1j * self.k_non_dimension * self.distance_non_dimension * torch.sqrt(deter_matrix))
        deter_matrix[mask] = 0
        deter_matrix = torch.fft.fftshift(deter_matrix)
        self.h = nn.Parameter(deter_matrix, requires_grad=False)

    def forward(self, x):
        # x (batch, 200, 200, 2)
        x = torch.fft.fft2(x)
        x = x * self.h
        # angular_spectrum (batch, 200, 200, 2)
        x = torch.fft.ifft2(x)
        return x


class ModulationLayer(nn.Module):
    def __init__(self, device, n_padd, original_size, scale_factor):
        super(ModulationLayer, self).__init__()
        self.scale_factor = scale_factor
        self.size = original_size  # 200 * 200 neurons in one layer
        self.padd = n_padd
        self.mod = nn.Parameter(torch.randn(self.size, self.size,
                                            requires_grad=True, dtype=torch.float32, device=device))

    def forward(self, x):
        mod_phase = torch.repeat_interleave(torch.repeat_interleave(self.mod, self.scale_factor, dim=0), self.scale_factor, dim=1)
        mod_phase = F.pad(torch.exp(1j * torch.pi * torch.sigmoid(mod_phase)),
                          pad=(self.padd, self.padd, self.padd, self.padd))
        x = x * mod_phase
        return x


class CutLayer(nn.Module):
    def __init__(self, n_padd, size, margin):
        super(CutLayer, self).__init__()
        self.start, self.end = n_padd + margin, (n_padd + size - margin)

    def forward(self, x):
        x = x[:, self.start:self.end, self.start:self.end]
        x = (x * torch.conj(x)).real
        return x


class IntensitySum(nn.Module):
    def __init__(self, disp, size, scale_factor):
        super(IntensitySum, self).__init__()
        self.N = size
        self.centers = [(19+disp, 19+disp), (19+disp, 49+disp), (19+disp, 81+disp),
                        (50+disp, 15+disp), (50+disp, 38+disp), (50+disp, 62+disp), (50+disp, 85+disp),
                        (81+disp, 19+disp), (81+disp, 49+disp), (81+disp, 81+disp)]
        gaussian_tensors = []
        for center in self.centers:
            center_n = tuple(x * scale_factor for x in center)
            gaussian_tensor = self.create_displaced_gaussian(center=center_n, sigma=3)
            gaussian_tensors.append(gaussian_tensor)
        self.gaussians = nn.Parameter(torch.stack(gaussian_tensors, dim=-1), requires_grad=False)

    def create_displaced_gaussian(self, center=None, sigma=10.0):
        if center is None:
            center = (self.N / 2, self.N / 2)  # Default to center, adjust this as needed
        x = torch.arange(self.N)
        y = torch.arange(self.N)

        xv, yv = torch.meshgrid(x, y, indexing='ij')
        gaussian = torch.exp(-((xv - center[0]) ** 2 + (yv - center[1]) ** 2) / (2 * sigma ** 2))

        return gaussian

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.expand(-1, -1, -1, 10)
        x = x * self.gaussians
        x = x.sum(dim=(1, 2))
        return x


class Dmodel(nn.Module):
    def __init__(self, pixel_pitch, distance, n_padd, device, lambd, size, margin, image_size, scale_factor):
        super(Dmodel, self).__init__()
        # self.start, self.end = n_padd, (n_padd + size)
        self.x_padd = (size-image_size)//2 + n_padd
        original_size = int(size/scale_factor)
        # self.n_layers = n_layers
        # Creates the convolution layers
        self.propagation = AngularPropagation(distance, n_padd, pixel_pitch, lambd, size)
        self.modulation_1 = ModulationLayer(device, n_padd, original_size, scale_factor)
        self.modulation_2 = ModulationLayer(device, n_padd, original_size, scale_factor)
        self.modulation_3 = ModulationLayer(device, n_padd, original_size, scale_factor)
        self.modulation_4 = ModulationLayer(device, n_padd, original_size, scale_factor)
        self.modulation_5 = ModulationLayer(device, n_padd, original_size, scale_factor)
        self.intensity_capture = CutLayer(n_padd, size, margin)
        self.intensity_sum = IntensitySum(50, size, scale_factor)

    def featurizer(self, x):
        x = torch.squeeze(x, dim=1)
        x = F.pad(x, pad=(self.x_padd, self.x_padd, self.x_padd, self.x_padd))
        x = self.propagation(x)
        x = self.modulation_1(x)
        x = self.propagation(x)
        x = self.modulation_2(x)
        x = self.propagation(x)
        x = self.modulation_3(x)
        x = self.propagation(x)
        x = self.modulation_4(x)
        x = self.propagation(x)
        x = self.modulation_5(x)
        x = self.propagation(x)
        x = self.intensity_capture(x)
        x = self.intensity_sum(x)
        return x

    def forward(self, x):
        x = self.featurizer(x)
        return x
