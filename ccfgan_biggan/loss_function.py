import torch
import torch.nn as nn
import os


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1
    """
    def __init__(self, alpha=1, beta=1):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, t, x, target):
        t_x = torch.mm(t, x.t())
        t_x_real = calculate_real(t_x)
        t_x_imag = calculate_imag(t_x)
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target = torch.mm(t, target.t())
        t_target_real = calculate_real(t_target)
        t_target_imag = calculate_imag(t_target)
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(t_target_norm, t_x_norm) -
                        torch.mul(t_x_real, t_target_real) -
                        torch.mul(t_x_imag, t_target_imag))

        loss_pha = loss_pha.clamp(min=1e-12)  # keep numerical stability

        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss
