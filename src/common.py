import torch
import torch.nn

from UNet.UNet import UNet
from UNet.ResUNet import ResUNetOnFreq, ResUNet

def get_model(hp):

    if hp.model.mag_only : 
        c_in = 1
        c_out = 1
    else :
        c_in = 2
        c_out = 2

    if hp.model.type == "UNet": 
        model = UNet(
        ).to(device)
    elif hp.model.type == "ResUNetOnFreq" :
        model = ResUNetOnFreq(c_in=c_in,c_out=c_out,n_fft=hp.data.n_fft,n_block=4).to(device)

    return model

def run(
    hp,
    data,
    model,
    criterion=None,
    ret_output=False,
    device="cuda:0"
    ): 
    feature = data["input"].to(device)
    mask = model(feature)

    # masking
    if hp.model.mag_only : 
        noisy_phase = data["noisy_phase"].to(device)
        estim_mag = feature*mask
        estim_spec = estim_mag * (noisy_phase*1j).to(device)
    else :
        estim_spec = mask * feature
        estim_spec = estim_spec[:,0:1,:,:]+estim_spec[:,1:2,:,:]*1j

    if criterion is None : 
        return estim_spec

    if hp.loss.type =="wSDRLoss" :
        estim_wav = torch.istft(estim_spec[:,0,:,:],n_fft = hp.data.n_fft)
        loss = criterion(estim_wav,data["noisy_wav"].to(device),data["clean_wav"].to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)
    elif hp.loss.type == "mwMSELoss" : 
        loss = criterion(estim_spec,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device)

    if ret_output :
        return estim_spec, loss
    else : 
        return loss
