import torch
import torch.nn as nn
import os
import math


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)

def calculate_ygx(y_pre, t_shape, y_embed, y_embed_out_ch, ty_lambda):  #calculate real and imaginary part of fai(y|x)
    batch_size, n_classes = y_pre.shape
    t2 = torch.linspace(-1, 1, steps=t_shape, device='cuda')
    t2 = t2.repeat(y_embed_out_ch, 1)
    ty = torch.mm(y_embed, t2)# calculate t2.t()·y
    ty = ty * ty_lambda
    cos_ty = torch.cos(ty)
    sin_ty = torch.sin(ty)
    Re = torch.mm(y_pre, cos_ty)
    Im = torch.mm(y_pre, sin_ty)
    return Re, Im

def calculate_fai(out, ygx_re, ygx_im): #calculate real and imaginary part of fai(x,y)
    Re = ygx_re * torch.cos(out) - ygx_im * torch.sin(out)
    Im = ygx_re * torch.sin(out) + ygx_im * torch.cos(out)
    Re = torch.mean(Re, dim=0, keepdim=True)
    Im = torch.mean(Im, dim=0, keepdim=True)
    Norm = calculate_norm(Re, Im)
    return Re, Im, Norm


class CFLossFuncCond(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1

    """

    def __init__(self, alpha=0.5, beta=0.5):
        super(CFLossFuncCond, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, target, y_x_pre, y_target_pre, y_embed, y_embed_out_ch, ty_lambda):
        t_shape = x.shape[1]
        x_ygx_re, x_ygx_im = calculate_ygx(y_x_pre, t_shape, y_embed, y_embed_out_ch, ty_lambda)
        target_ygx_re, target_ygx_im = calculate_ygx(y_target_pre, t_shape, y_embed, y_embed_out_ch, ty_lambda)
        x_Re, x_Im, x_Norm = calculate_fai(x, x_ygx_re, x_ygx_im)
        target_Re, target_Im, target_Norm = calculate_fai(target, target_ygx_re, target_ygx_im)

        amp_diff = target_Norm - x_Norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(target_Norm, x_Norm) -
                        torch.mul(x_Re, target_Re) -
                        torch.mul(x_Im, target_Im))

        loss_amp = loss_amp.clamp(min=1e-12)  # keep numerical stability
        loss_pha = loss_pha.clamp(min=1e-12)  # keep numerical stability

        loss = torch.sqrt(torch.mean(self.alpha * loss_amp + self.beta * loss_pha))
        return loss