import torch
import torch.nn as nn
import math


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


def euler2quaternion2d(euler_angles):
    """Operates on an array of Euler angles that have been normalised to the range [0, 1].

    Parameters
    ----------
    euler_angles : torch.Tensor
        Tensor of normalised Euler angles. Should have shape (batch_size, 3, patch_size, patch_size).

    Returns
    -------
    quaternions : torch.Tensor
        Tensor of quaterions with shape (batch_size, 4, patch_size, patch_size).
    """

    quaternions = torch.zeros((euler_angles.shape[0], 4, euler_angles.shape[2], euler_angles.shape[3]))

    psi = euler_angles[:, 0, :, :] * torch.tensor(math.pi)
    theta = (euler_angles[:, 1, :, :] - (0.5 * torch.tensor(math.pi))) * 0.5 * torch.tensor(math.pi)
    phi = euler_angles[:, 2, :, :] * torch.tensor(math.pi)

    q_w = (torch.cos(phi) * torch.cos(theta) * torch.cos(psi)) + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi))
    q_x = (torch.sin(phi) * torch.cos(theta) * torch.cos(psi)) - (torch.cos(phi) * torch.sin(theta) * torch.sin(psi))
    q_y = (torch.cos(phi) * torch.sin(theta) * torch.cos(psi)) + (torch.sin(phi) * torch.cos(theta) * torch.sin(psi))
    q_z = (torch.cos(phi) * torch.cos(theta) * torch.sin(psi)) - (torch.sin(phi) * torch.sin(theta) * torch.cos(psi))

    quaternions[:, 0, :, :] = q_w
    quaternions[:, 1, :, :] = q_x
    quaternions[:, 2, :, :] = q_y
    quaternions[:, 3, :, :] = q_z

    return quaternions


def euler2quaternion3d(euler_angles):
    """Operates on an array of Euler angles that have been normalised to the range [0, 1].

    Parameters
    ----------
    euler_angles : torch.Tensor
        Tensor of normalised Euler angles. Should have shape (batch_size, 3, patch_size, patch_size, patch_size).

    Returns
    -------
    quaternions : torch.Tensor
        Tensor of quaterions with shape (batch_size, 4, patch_size, patch_size, patch_size).
    """

    quaternions = torch.zeros((euler_angles.shape[0], 4, euler_angles.shape[2], euler_angles.shape[3], euler_angles.shape[4]))

    psi = euler_angles[:, 0, :, :, :] * torch.tensor(math.pi)
    theta = (euler_angles[:, 1, :, :, :] - (0.5 * torch.tensor(math.pi))) * 0.5 * torch.tensor(math.pi)
    phi = euler_angles[:, 2, :, :, :] * torch.tensor(math.pi)

    q_w = (torch.cos(phi) * torch.cos(theta) * torch.cos(psi)) + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi))
    q_x = (torch.sin(phi) * torch.cos(theta) * torch.cos(psi)) - (torch.cos(phi) * torch.sin(theta) * torch.sin(psi))
    q_y = (torch.cos(phi) * torch.sin(theta) * torch.cos(psi)) + (torch.sin(phi) * torch.cos(theta) * torch.sin(psi))
    q_z = (torch.cos(phi) * torch.cos(theta) * torch.sin(psi)) - (torch.sin(phi) * torch.sin(theta) * torch.cos(psi))

    quaternions[:, 0, :, :, :] = q_w
    quaternions[:, 1, :, :, :] = q_x
    quaternions[:, 2, :, :, :] = q_y
    quaternions[:, 3, :, :, :] = q_z

    return quaternions
