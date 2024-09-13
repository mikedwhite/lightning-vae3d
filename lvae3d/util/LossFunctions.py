import torch
import torch.nn as nn
from torch.fft import fftn, fftshift
import damask

import math

from lvae3d.util.Mappings import euler_distance, eu2qu2d, eu2qu3d


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, log_sigma):
        """Computes the KL divergence between the latent space and a unit Gaussian distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Vector of means used to construct the fingerprint.
        log_sigma : torch.Tenor
            Vector of log standard deviations used to construct the fingerprint.

        Returns
        -------
        loss : torch.Tensor
            KL divergence.
        """

        loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return loss


class SpectralLoss2D(nn.Module):
    def __init__(self):
        super(SpectralLoss2D, self).__init__()

    def forward(self, x, x_hat):
        r"""Computes the mean squared error between the 2D Fourier transforms of the input image, :math:`x`, and the
        corresponding reconstruction, :math:`\hat{x}`.
        Must be applied to image tensors with shape (`n_images`, `h`, `w`, `c`), where `n_images` is the number of
        images in the batch, `h` and `w` are the height and width of the image, respectively, and `c` is the number of
        channels in the image.

        Parameters
        ----------
        x : torch.Tensor
            Input image with shape (n_images, h, w, c).
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Spectral loss between the input and the reconstruction.
        """
        n_images = x.shape[0]
        n_pixels = x.shape[-2] * x.shape[-1]
        fft_x, fft_x_hat = fftshift(fftn(x, dim=(-2, -1))), fftshift(fftn(x_hat, dim=(-2, -1)))
        loss = (1 / (n_pixels * n_images)) * \
                (((fft_x.imag - fft_x_hat.imag) ** 2).sum() + ((fft_x.real - fft_x_hat.real) ** 2).sum())
        return loss


class SpectralLoss3D(nn.Module):
    def __init__(self):
        super(SpectralLoss3D, self).__init__()

    def forward(self, x, x_hat):
        r"""Computes the mean squared error between the 3D Fourier transforms of the input image, :math:`x`, and the
        corresponding reconstruction, :math:`\hat{x}`.
        Must be applied to image tensors with shape (`n_images`, `h`, `w`, `d`, `c`), where `n_images` is the number of
        images in the batch, `h`, `w` and `d` are the height, width and depth of the image, respectively, and `c` is the
        number of channels in the image.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Spectral loss between the input and the reconstruction.
        """
        n_images = x.shape[0]
        n_pixels = x.shape[-3] * x.shape[-2] * x.shape[-1]
        fft_x, fft_x_hat = fftshift(fftn(x, dim=(-3, -2, -1))), fftshift(fftn(x_hat, dim=(-3, -2, -1)))
        loss = (1 / (n_pixels * n_images)) * \
                (((fft_x.imag - fft_x_hat.imag) ** 2).sum() + ((fft_x.real - fft_x_hat.real) ** 2).sum())
        return loss


class EulerLoss2D(nn.Module):
    def __init__(self):
        super(EulerLoss2D, self).__init__()

    def forward(self, x, x_hat):
        """Uses phi_1 in 'Metrics for 3D Rotations: Comparison and Analysis', Du Q. Huynh.
        Operates on 2D slices of a volume element.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Euler loss between the input and the reconstruction.
        """
        alpha1 = math.pi * (x[:, 0, :, :] - 1)
        alpha2 = math.pi * (x_hat[:, 0, :, :] - 1)
        beta1 = math.pi * (x[:, 1, :, :] - 1) * 0.5
        beta2 = math.pi * (x_hat[:, 1, :, :] - 1) * 0.5
        gamma1 = math.pi * (x[:, 2, :, :] - 1)
        gamma2 = math.pi * (x_hat[:, 2, :, :] - 1)
        distance = torch.sqrt(euler_distance(alpha1, alpha2) ** 2 + \
                              euler_distance(beta1, beta2) ** 2 + \
                              euler_distance(gamma1, gamma2) ** 2)
        loss = torch.linalg.vector_norm(torch.flatten(distance))
        return loss


class EulerLoss3D(nn.Module):
    def __init__(self):
        super(EulerLoss3D, self).__init__()

    def forward(self, x, x_hat):
        """Uses phi_1 in 'Metrics for 3D Rotations: Comparison and Analysis', Du Q. Huynh.
        Operates on a 3D volume element.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Euler loss between the input and the reconstruction.
        """
        alpha1 = math.pi * (x[:, 0, :, :, :] - 1)
        alpha2 = math.pi * (x_hat[:, 0, :, :, :] - 1)
        beta1 = math.pi * (x[:, 1, :, :, :] - 1) * 0.5
        beta2 = math.pi * (x_hat[:, 1, :, :, :] - 1) * 0.5
        gamma1 = math.pi * (x[:, 2, :, :, :] - 1)
        gamma2 = math.pi * (x_hat[:, 2, :, :, :] - 1)
        distance = torch.sqrt(euler_distance(alpha1, alpha2) ** 2 + \
                              euler_distance(beta1, beta2) ** 2 + \
                              euler_distance(gamma1, gamma2) ** 2)
        loss = torch.linalg.vector_norm(torch.flatten(distance))
        return loss


class QuaternionLoss2D(nn.Module):
    def __init__(self):
        super(QuaternionLoss2D, self).__init__()

    def forward(self, x, x_hat):
        """Rotational distance from 'Q-RBSA: high-resolution 3D EBSD map generation using an efficient quaternion
        transformer network', D. K. Jangit et al.
        Operates on a 2D volume element.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Quaternion loss between the input and the reconstruction.
        """
        q, q_hat = eu2qu2d(x), eu2qu2d(x_hat)
        theta = 4 * torch.asin(torch.sqrt(torch.sum(torch.mul(q - q_hat, q - q_hat), axis=1) / 2))
        loss = torch.linalg.norm(torch.flatten(theta), ord=2)
        return loss


class QuaternionLoss3D(nn.Module):
    def __init__(self):
        super(QuaternionLoss3D, self).__init__()

    def forward(self, x, x_hat):
        """Rotational distance from 'Q-RBSA: high-resolution 3D EBSD map generation using an efficient quaternion
        transformer network', D. K. Jangit et al.
        Operates on a 3D volume element.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Quaternion loss between the input and the reconstruction.
        """
        q, q_hat = eu2qu3d(x), eu2qu3d(x_hat)
        theta = 4 * torch.asin(torch.sqrt(torch.sum(torch.mul(q - q_hat, q - q_hat), axis=1)) / 2)
        loss = torch.linalg.norm(torch.flatten(theta), ord=2)
        return loss


class EulerMisorientation3D(nn.Module):
    def __init__(self):
        super(EulerMisorientation3D, self).__init__()

    def forward(self, x, x_hat):
        eu, eu_hat = torch.moveaxis(x, 0, -1), torch.moveaxis(x_hat, 0, -1)
        eu = torch.reshape(eu, (-1, 3)) * torch.tensor(([2.0 * math.pi, math.pi, 2.0 * math.pi]))
        eu_hat = torch.reshape(eu_hat, (-1, 3)) * torch.tensor(([2.0 * math.pi, math.pi, 2.0 * math.pi]))

        misorientation = torch.zeros(eu.shape[0])
        for n in range(eu.shape[0]):
            g = torch.zeros((3, 3))
            g_hat = torch.zeros((3, 3))

            g[0, 0] = torch.cos(eu[n, 0]) * torch.cos(eu[n, 2]) - torch.sin(eu[n, 0]) * torch.sin(eu[n, 2]) * torch.cos(eu[n, 1])
            g[0, 1] = torch.sin(eu[n, 0]) * torch.cos(eu[n, 2]) + torch.cos(eu[n, 0]) * torch.sin(eu[n, 2]) * torch.cos(eu[n, 1])
            g[0, 2] = torch.sin(eu[n, 2]) * torch.sin(eu[n, 1])
            g[1, 0] = -torch.cos(eu[n, 0]) * torch.sin(eu[n, 2]) - torch.sin(eu[n, 0]) * torch.cos(eu[n, 2]) * torch.cos(eu[n, 1])
            g[1, 1] = -torch.sin(eu[n, 0]) * torch.sin(eu[n, 2]) + torch.cos(eu[n, 0]) * torch.cos(eu[n, 2]) * torch.cos(eu[n, 1])
            g[1, 2] = torch.cos(eu[n, 2]) * torch.sin(eu[n, 1])
            g[2, 0] = torch.sin(eu[n, 0]) * torch.sin(eu[n, 1])
            g[2, 1] = -torch.cos(eu[n, 0]) * torch.sin(eu[n, 1])
            g[2, 2] = torch.cos(eu[n, 1])

            g_hat[0, 0] = torch.cos(eu_hat[n, 0]) * torch.cos(eu_hat[n, 2]) - torch.sin(eu_hat[n, 0]) * torch.sin(eu_hat[n, 2]) * torch.cos(eu_hat[n, 1])
            g_hat[0, 1] = torch.sin(eu_hat[n, 0]) * torch.cos(eu_hat[n, 2]) + torch.cos(eu_hat[n, 0]) * torch.sin(eu_hat[n, 2]) * torch.cos(eu_hat[n, 1])
            g_hat[0, 2] = torch.sin(eu_hat[n, 2]) * torch.sin(eu_hat[n, 1])
            g_hat[1, 0] = -torch.cos(eu_hat[n, 0]) * torch.sin(eu_hat[n, 2]) - torch.sin(eu_hat[n, 0]) * torch.cos(eu_hat[n, 2]) * torch.cos(eu_hat[n, 1])
            g_hat[1, 1] = -torch.sin(eu_hat[n, 0]) * torch.sin(eu_hat[n, 2]) + torch.cos(eu_hat[n, 0]) * torch.cos(eu_hat[n, 2]) * torch.cos(eu_hat[n, 1])
            g_hat[1, 2] = torch.cos(eu_hat[n, 2]) * torch.sin(eu_hat[n, 1])
            g_hat[2, 0] = torch.sin(eu_hat[n, 0]) * torch.sin(eu_hat[n, 1])
            g_hat[2, 1] = -torch.cos(eu_hat[n, 0]) * torch.sin(eu_hat[n, 1])
            g_hat[2, 2] = torch.cos(eu_hat[n, 1])

            g_diff = g_hat * torch.linalg.inv(g)
            misorientation[n] = torch.acos(0.5 * (g_diff[0, 0] + g_diff[1, 1] + g_diff[2, 2] - 1)) ** 2

        loss = torch.mean(misorientation)

        return loss
