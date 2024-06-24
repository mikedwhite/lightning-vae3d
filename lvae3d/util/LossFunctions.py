import torch
import torch.nn as nn
from torch.fft import fftn, fftshift

import math


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

        loss = (mu ** 2 + torch.exp(log_sigma) ** 2 - log_sigma - 0.5).sum()
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
        n_pixels = x.shape[-2] * x.shape[-1]
        fft_x, fft_x_hat = fftshift(fftn(x, dim=(-3, -2, -1))), fftshift(fftn(x_hat, dim=(-3, -2, -1)))
        loss = (1 / (n_pixels * n_images)) * \
                (((fft_x.imag - fft_x_hat.imag) ** 2).sum() + ((fft_x.real - fft_x_hat.real) ** 2).sum())
        return loss


def euler_distance(a, b):
    """Compute the element-wise distance between two 2D tensors of Euler angles, `a` and `b`. `a` and `b` must have the
    same shape.

    Parameters
    ----------
    a : torch.Tensor
        Tensor of Euler angles.
    b : torch.Tensor
        Tensor of Euler angles.

    Returns
    -------
    distance : torch.Tensor
        Tensor containing the element-wise distance between `a` and `b`.
    """
    distance = torch.minimum(torch.abs(a - b), 2 * torch.Tensor([math.pi]).type_as(a) - torch.abs(a - b))
    return distance


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
        mean_distance = torch.linalg.vector_norm(torch.flatten(distance))
        return mean_distance


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
        mean_distance = torch.linalg.vector_norm(torch.flatten(distance))
        return mean_distance
