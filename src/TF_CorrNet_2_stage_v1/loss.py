import torch
import numpy as np

from math import ceil
from itertools import permutations
from dataclasses import dataclass, field, fields
from loguru import logger
from utils.decorators import *
from utils import util_stft

# Utility functions
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)


def filtering(filter, mixture):
	# filter : B, W, F, T
	# mixture : B, F, T
	B, M, W, F, T = filter.shape

	# padding for temporal deep filtering
	kernel = torch.eye(W) + 0.0j
	kernel = kernel.reshape(W,1,1,W).to(filter.device)
	mix = mixture.view(B*M, F, T)
	mix_padded = torch.nn.functional.pad(mix[:,None], [(W-1)//2,(W-1)//2]) # B, M, 1, F, T
	mix_padded = torch.nn.functional.conv2d(mix_padded, kernel) # B, M, W, F, T
	mix_padded = mix_padded.contiguous().view(B, M, W, F, T) # B, M, W, F, T

	# channel filter and sum
	return torch.einsum("...mwft,...mwft->...ft",[mix_padded, filter]) # B, F, T


@logger_wraps()
class PIT_MSE_complex(torch.nn.Module):
	def __init__(self, num_spks, device, ref_ch=0):
		super().__init__()
		self.num_spks = num_spks
		self.device = device
		self.ref_ch = ref_ch


	def forward(self, masks, input_sizes, mixture_stft, target_stft):
		input_sizes = input_sizes.to(self.device)
		masks = [torch.complex(mask[...,0], mask[...,1]) for mask in masks]  # [M, F, T]
		mixture = [mix.to(self.device) for mix in mixture_stft]
		targets = [t[:,self.ref_ch].to(self.device) for t in target_stft] # [B, F, T]

		def A_RI_loss(permute, eps=1.0e-6):
			loss_m = loss_r = loss_i = []
			for s, t in enumerate(permute):
				out = filtering(masks[s],mixture[s])
				src = targets[t]
				loss_r.append(l1norm(l1norm(src.real-out.real)))
				loss_i.append(l1norm(l1norm(src.imag-out.imag)))
				loss_m.append(l1norm(l1norm(src.abs()-out.abs())))

			RI_loss = sum(loss_r) + sum(loss_i)
			Mag_loss = sum(loss_m)
			return (RI_loss + Mag_loss) / self.num_spks

		pscore = torch.stack( [ A_RI_loss(p) for p in permutations(range(self.num_spks))] )
		min_perutt, min_idx = torch.min(pscore, dim=0)
		num_utts = input_sizes.shape[0]

		return torch.sum(min_perutt) / num_utts

@logger_wraps()
class PIT_MSE_time(torch.nn.Module):
	def __init__(self, frame_length, frame_shift, num_spks, device, ref_ch=0):
		super().__init__()
		self.num_spks = num_spks
		self.device = device
		self.istft = util_stft.iSTFT(frame_length, frame_shift, device=device)
		self.frame_shift =frame_shift
		self.ref_ch = ref_ch

	def forward(self, masks, input_sizes, mixture_stft, target_stft):
		input_sizes = input_sizes.to(self.device)
		masks = [torch.complex(mask[...,0], mask[...,1]) for mask in masks]  # [M, F, T]
		mixture = [mix.to(self.device) for mix in mixture_stft]
		targets = [t[:,self.ref_ch].to(self.device) for t in target_stft] # [B, F, T]

		def A_Time_loss(permute, eps=1.0e-10):
			loss_t = []
			for s, t in enumerate(permute):
				out = filtering(masks[s],mixture[s])
				out = self.istft(out, cplx=True)
				src = self.istft(targets[t], cplx=True)
				loss_t.append(l1norm(src-out))

			Time_loss = sum(loss_t)
			return Time_loss / self.num_spks

          
		pscore = torch.stack( [ A_Time_loss(p) for p in permutations(range(self.num_spks))] )
		min_perutt, min_idx = torch.min(pscore, dim=0)
		num_utts = input_sizes.shape[0]

		return torch.sum(min_perutt) / num_utts


@logger_wraps()
class MC_MSE_time(torch.nn.Module):
	def __init__(self, frame_length, frame_shift, num_spks, device, ref_ch=0):
		super().__init__()
		self.num_spks = num_spks
		self.device = device
		self.istft = util_stft.iSTFT(frame_length, frame_shift, device=device)
		self.frame_shift =frame_shift
		self.ref_ch = ref_ch


	def forward(self, masks, input_sizes, mixture_stft, target_stft):
		input_sizes = input_sizes.to(self.device)
		masks = [torch.complex(mask[...,0], mask[...,1]) for mask in masks]  # [M, F, T]
		mixture = [mix.to(self.device) for mix in mixture_stft]
		targets = [t[:,self.ref_ch].to(self.device) for t in target_stft] # [B, F, T]
  
		def MC_Time_loss(eps=1.0e-10):
			out_list = []
			src_list = []
			for i in range(self.num_spks):
				out = filtering(masks[i],mixture[i])
				out = self.istft(out, cplx=True)
				out_list.append(out)
				src = self.istft(targets[i], cplx=True)
				src_list.append(src)
			out_mixture = sum(out_list)
			src_mixture = sum(src_list)
			loss = l1norm(src_mixture-out_mixture)
			return sum(loss)

		num_utts = input_sizes.shape[0]

		return MC_Time_loss() / num_utts