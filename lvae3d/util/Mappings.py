import torch
import torch.nn as nn
import math


_P = -1


def euler_distance(a, b):
    """Compute the element-wise distance between two tensors of Euler angles, `a` and `b`. `a` and `b` must have the
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
    # distance = torch.minimum(torch.abs(a - b), 2 * torch.Tensor([math.pi]).type_as(a) - torch.abs(a - b))
    distance = torch.sin(a - b) ** 2
    return distance


def eu2qu2d(eu):
    """Operates on a tensor of Euler angles that have been normalised to the range [0, 1].

    Parameters
    ----------
    eu : torch.Tensor
        Tensor of normalised Euler angles. Should have shape (batch_size, 3, patch_size, patch_size).

    Returns
    -------
    qu : torch.Tensor
        Tensor of quaterions with shape (batch_size, 4, patch_size, patch_size).
    """

    qu = torch.zeros((eu.shape[0], 4, eu.shape[2], eu.shape[3]))

    psi = eu[:, 0, :, :] * torch.tensor(math.pi)
    theta = (eu[:, 1, :, :] - (0.5 * torch.tensor(math.pi))) * 0.5 * torch.tensor(math.pi)
    phi = eu[:, 2, :, :] * torch.tensor(math.pi)

    q_w = (torch.cos(phi) * torch.cos(theta) * torch.cos(psi)) + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi))
    q_x = (torch.sin(phi) * torch.cos(theta) * torch.cos(psi)) - (torch.cos(phi) * torch.sin(theta) * torch.sin(psi))
    q_y = (torch.cos(phi) * torch.sin(theta) * torch.cos(psi)) + (torch.sin(phi) * torch.cos(theta) * torch.sin(psi))
    q_z = (torch.cos(phi) * torch.cos(theta) * torch.sin(psi)) - (torch.sin(phi) * torch.sin(theta) * torch.cos(psi))

    qu[:, 0, :, :] = q_w
    qu[:, 1, :, :] = q_x
    qu[:, 2, :, :] = q_y
    qu[:, 3, :, :] = q_z

    qu = torch.moveaxis(qu, 0, 1)
    qu[:, qu[0, :, :, :] < 0.0] *= -1
    qu = torch.moveaxis(qu, 1, 0)

    return qu


def eu2qu3d(eu):
    """Operates on a tensor of Euler angles that have been normalised to the range [0, 1].

    Parameters
    ----------
    eu : torch.Tensor
        Tensor of normalised Euler angles. Should have shape (batch_size, 3, patch_size, patch_size, patch_size).

    Returns
    -------
    qu : torch.Tensor
        Tensor of quaterions with shape (batch_size, 4, patch_size, patch_size, patch_size).
    """

    qu = torch.zeros((eu.shape[0], 4, eu.shape[2], eu.shape[3], eu.shape[4]))

    psi = eu[:, 0, :, :, :] * 2.0 * torch.tensor(math.pi)
    theta = eu[:, 1, :, :, :] * torch.tensor(math.pi)
    phi = eu[:, 2, :, :, :] * 2.0 * torch.tensor(math.pi)

    sigma = 0.5 * (psi + phi)
    delta = 0.5 * (psi - phi)
    c = torch.cos(0.5 * theta)
    s = torch.sin(0.5 * theta)

    qu[:, 0, :, :, :] = c * torch.cos(sigma)
    qu[:, 1, :, :, :] = -_P * s * torch.cos(delta)
    qu[:, 2, :, :, :] = -_P * s * torch.sin(delta)
    qu[:, 3, :, :, :] = -_P * c * torch.sin(sigma)

    qu = torch.moveaxis(qu, 0, 1)
    qu[:, qu[0, :, :, :, :] < 0.0] *= -1
    qu = torch.moveaxis(qu, 1, 0)

    return qu
