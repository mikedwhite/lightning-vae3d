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
        loss = torch.mul(torch.div(1, torch.mul(n_pixels, n_images)),
                         torch.sum((fft_x.imag - fft_x_hat.imag) ** 2) + torch.sum((fft_x.real - fft_x_hat.real) ** 2))
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
        n_voxels = x.shape[-3] * x.shape[-2] * x.shape[-1]
        fft_x, fft_x_hat = fftshift(fftn(x, dim=(-3, -2, -1))), fftshift(fftn(x_hat, dim=(-3, -2, -1)))
        loss = torch.mul(torch.div(1, torch.mul(n_voxels, n_images)),
                         torch.sum((fft_x.imag - fft_x_hat.imag) ** 2) + torch.sum((fft_x.real - fft_x_hat.real) ** 2))
        return loss


class QuaternionMisorientation3Dqu(nn.Module):
    def __init__(self):
        super(QuaternionMisorientation3Dqu, self).__init__()

    def forward(self, x, x_hat):
        """Computes the mean squared error of minimum misorientation between symmetric equivalents of quaternions.
        Operates on a batch of 3D tensors of normalised quaternions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Mean squared error of minimum quaternion misorientation between the input and the reconstruction.
        """
        symmetry_ops = torch.tensor(([[ 1.0,                             0.0,                             0.0,                             0.0                             ],
                                      [ 0.0,                             1.0,                             0.0,                             0.0                             ],
                                      [ 0.0,                             0.0,                             1.0,                             0.0                             ],
                                      [ 0.0,                             0.0,                             0.0,                             1.0                             ],
                                      [ 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.0,                            -0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [ 0.0,                            -0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [ 0.5,                             0.5,                             0.5,                             0.5                             ],
                                      [-0.5,                             0.5,                             0.5,                             0.5                             ],
                                      [-0.5,                             0.5,                             0.5,                            -0.5                             ],
                                      [-0.5,                             0.5,                            -0.5,                             0.5                             ],
                                      [-0.5,                            -0.5,                             0.5,                             0.5                             ],
                                      [-0.5,                            -0.5,                             0.5,                            -0.5                             ],
                                      [-0.5,                            -0.5,                            -0.5,                             0.5                             ],
                                      [-0.5,                             0.5,                            -0.5,                            -0.5                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)) ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.0,                            -0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0                             ],
                                      ])).type_as(x)

        q, q_hat = torch.moveaxis(x, 0, -1), torch.moveaxis(x_hat, 0, -1)
        q = torch.reshape(q, (4, -1))
        q_hat = torch.reshape(q_hat, (4, -1))

        q_syms = torch.empty(symmetry_ops.shape[0], q.shape[0], q.shape[1]).type_as(q)
        for n, sym in enumerate(symmetry_ops):
            q_syms[n, 0, :] = q[0, :]*sym[0] - q[1, :]*sym[1] - q[2, :]*sym[2] - q[3, :]*sym[3]
            q_syms[n, 1, :] = q[0, :]*sym[1] + q[1, :]*sym[0] - q[2, :]*sym[3] + q[3, :]*sym[2]
            q_syms[n, 2, :] = q[0, :]*sym[2] + q[2, :]*sym[0] - q[3, :]*sym[1] + q[1, :]*sym[3]
            q_syms[n, 3, :] = q[0, :]*sym[3] + q[3, :]*sym[0] - q[1, :]*sym[2] + q[2, :]*sym[1]

        args = torch.argwhere(q_syms[:, 0, :] < 0.0)  # check if this is needed (symmetric variants should stay in northern hemisphere)
        q_syms[args[:, 0], :, args[:, 1]] *= -1

        for n in range(q_syms.shape[0]):
            if n == 0:
                min_misorientation = torch.abs(torch.acos(torch.abs(torch.sum(torch.mul(q_hat, q_syms[n, :, :]), axis=0))))
            else:
                temp_misorientation = torch.abs(torch.acos(torch.abs(torch.sum(torch.mul(q_hat, q_syms[n, :, :]), axis=0))))
                min_misorientation = torch.where(temp_misorientation < min_misorientation, temp_misorientation, min_misorientation)

        loss = torch.sum(torch.mul(min_misorientation, min_misorientation)) / min_misorientation.shape[0]

        return loss


class QuaternionMisorientation3Deu(nn.Module):
    def __init__(self):
        super(QuaternionMisorientation3Deu, self).__init__()

    def forward(self, x, x_hat):
        """Computes the mean squared error of minimum misorientation between symmetric equivalents of quaternions.
        Operates on a batch of 3D tensors of normalised Euler angles.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Mean squared error of minimum quaternion misorientation between the input and the reconstruction.
        """
        symmetry_ops = torch.tensor(([[ 1.0,                             0.0,                             0.0,                             0.0                             ],
                                      [ 0.0,                             1.0,                             0.0,                             0.0                             ],
                                      [ 0.0,                             0.0,                             1.0,                             0.0                             ],
                                      [ 0.0,                             0.0,                             0.0,                             1.0                             ],
                                      [ 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.0,                            -0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.0,                             0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [ 0.0,                            -0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [ 0.5,                             0.5,                             0.5,                             0.5                             ],
                                      [-0.5,                             0.5,                             0.5,                             0.5                             ],
                                      [-0.5,                             0.5,                             0.5,                            -0.5                             ],
                                      [-0.5,                             0.5,                            -0.5,                             0.5                             ],
                                      [-0.5,                            -0.5,                             0.5,                             0.5                             ],
                                      [-0.5,                            -0.5,                             0.5,                            -0.5                             ],
                                      [-0.5,                            -0.5,                            -0.5,                             0.5                             ],
                                      [-0.5,                             0.5,                            -0.5,                            -0.5                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)) ],
                                      [ 0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0,                             0.5*torch.sqrt(torch.tensor(2)) ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.0,                            -0.5*torch.sqrt(torch.tensor(2)), 0.0                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)), 0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0                             ],
                                      [-0.5*torch.sqrt(torch.tensor(2)),-0.5*torch.sqrt(torch.tensor(2)), 0.0,                             0.0                             ],
                                      ])).type_as(x)

        q, q_hat = eu2qu3d(x).type_as(x), eu2qu3d(x_hat).type_as(x_hat)
        q, q_hat = torch.moveaxis(q, 0, -1), torch.moveaxis(q_hat, 0, -1)
        q = torch.reshape(q, (4, -1))
        q_hat = torch.reshape(q_hat, (4, -1))

        q_syms = torch.empty(symmetry_ops.shape[0], q.shape[0], q.shape[1]).type_as(q)
        for n, sym in enumerate(symmetry_ops):
            q_syms[n, 0, :] = q[0, :]*sym[0] - q[1, :]*sym[1] - q[2, :]*sym[2] - q[3, :]*sym[3]
            q_syms[n, 1, :] = q[0, :]*sym[1] + q[1, :]*sym[0] - q[2, :]*sym[3] + q[3, :]*sym[2]
            q_syms[n, 2, :] = q[0, :]*sym[2] + q[2, :]*sym[0] - q[3, :]*sym[1] + q[1, :]*sym[3]
            q_syms[n, 3, :] = q[0, :]*sym[3] + q[3, :]*sym[0] - q[1, :]*sym[2] + q[2, :]*sym[1]

        args = torch.argwhere(q_syms[:, 0, :] < 0.0)  # check if this is needed (symmetric variants should stay in northern hemisphere)
        q_syms[args[:, 0], :, args[:, 1]] *= -1

        for n in range(q_syms.shape[0]):
            if n == 0:
                min_misorientation = torch.abs(torch.acos(torch.abs(torch.sum(torch.mul(q_hat, q_syms[n, :, :]), axis=0))))
            else:
                temp_misorientation = torch.abs(torch.acos(torch.abs(torch.sum(torch.mul(q_hat, q_syms[n, :, :]), axis=0))))
                min_misorientation = torch.where(temp_misorientation < min_misorientation, temp_misorientation, min_misorientation)

        loss = torch.sum(torch.mul(min_misorientation, min_misorientation)) / min_misorientation.shape[0]

        return loss


class EulerMisorientation(nn.Module):
    def __init__(self):
        super(EulerMisorientation, self).__init__()

    def forward(self, x, x_hat):
        """Computes the mean squared error of misorientation between Euler angles.
        Operates on a batch of tensors of normalised Euler angles. Can be 3D or 2D.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x_hat : torch.Tensor
            Reconstruction.

        Returns
        -------
        loss : torch.Tensor
            Mean squared error of Euler angle misorientation between the input and the reconstruction.
        """
        eu, eu_hat = torch.moveaxis(x, 0, -1), torch.moveaxis(x_hat, 0, -1)
        eu = torch.reshape(eu, (3, -1))
        eu = torch.transpose(eu, 0, 1) * torch.tensor(([2.0 * math.pi, math.pi, 2.0 * math.pi])).type_as(eu)
        eu_hat = torch.reshape(eu_hat, (3, -1))
        eu_hat = torch.transpose(eu_hat, 0, 1) * torch.tensor(([2.0 * math.pi, math.pi, 2.0 * math.pi])).type_as(eu_hat)

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
