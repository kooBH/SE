import torch
import torch.nn

from UNet.UNet import UNet
from UNet.ResUNet import ResUNetOnFreq, ResUNet, ResUNetOnFreq2
from FSN.FullSubNet_Plus import FullSubNet_Plus
from mpSE.TRUNet import TRUNet
from MTFAA.MTFAA import MTFAA_helper
import librosa as rs

def get_model(hp,device="cuda:0"):
    if hp.model.type == "UNet": 
        model = UNet().to(device)
    elif hp.model.type == "ResUNetOnFreq" :
        model = ResUNetOnFreq(
            c_in=c_in,
            c_out=c_out,
            n_fft=hp.audio.n_fft,
            n_block=5,
            norm = hp.model.norm,
            Softplus_thr=hp.model.Softplus_thr,
            activation = hp.model.activation,
            dropout = hp.model.dropout
            ).to(device)
    elif hp.model.type == "ResUNetOnFreq2" :
        model = ResUNetOnFreq2(
            c_in=c_in,
            c_out=c_out,
            n_fft=hp.audio.n_fft,
            n_block=5,
            norm = hp.model.norm,
            Softplus_thr=hp.model.Softplus_thr,
            activation = hp.model.activation,
            dropout = hp.model.dropout,
            multi_scale=hp.model.multi_scale
            ).to(device)
    elif hp.model.type == "FullSubNetPlus" : 
        model = FullSubNet_Plus(num_freqs = hp.model.n_freq).to(device)
    elif hp.model.type == "TRUMEA" : 
        from mpSE.TRUNet import TRUNet
        model = TRUNet(
            hp.audio.n_fft,
            hp.audio.n_hop,
            use_FSABlock=hp.model.use_FSABlock,
            architecture=hp.model.architecture,
            kernel_type = hp.model.kernel_type,
            skipGRU= hp.model.skipGRU,
            phase_encoder=hp.model.phase_encoder
         ).to(device)
    elif hp.model.type == "MTFAA" :
        model = MTFAA_helper(
            n_fft = hp.model.n_fft,
            n_hop = hp.model.n_hop,
            n_erb = hp.model.n_erb,
            Co = hp.model.Co,
            type_encoder = hp.model.type_encoder,
            type_ASA = hp.model.type_ASA
        ).to(device)
    else : 
        raise Exception("ERROR::Unknown model type : {}".format(hp.model.type))

    return model

def run(
    hp,
    data,
    model,
    criterion=None,
    ret_output=False,
    device="cuda:0"
    ): 

    if hp.model.type == "FullSubNetPlus":
        data["input"][0]=data["input"][0].to(device)
        data["input"][1]=data["input"][1].to(device)
        data["input"][2]=data["input"][2].to(device)
        feature = data["input"]
        mask = model(feature[0],feature[1],feature[2])
        estim= model.output(mask,feature[1],feature[2])
    elif hp.model.type == "ResUNetOnFreq" : 
        feature = data["noisy"].to(device)
        mask = model(feature)
        estim= model.output(mask,feature)
    elif hp.model.type == "ResUNetOnFreq2" : 
        feature = data["noisy"].to(device)
        mask = model(feature)
        estim= model.output(mask,feature)
    elif hp.model.type == "ResUNetOnFreq3" : 
        feature = data["noisy"].to(device)
        mask = model(feature)
        estim= model.output(mask,feature)
    elif hp.model.type =="TRUMEA" : 
        feature = data["noisy"].to(device)
        estim = model(feature)
    elif hp.model.type =="MTFAA" : 
        feature = data["noisy"].to(device)
        estim = model(feature)
    else : 
        raise Exception("ERROR::Unnkwon Model : {}".format(hp.model.type))

    if criterion is None : 
        return estim

    if hp.loss.type =="wSDRLoss" :
        clean= data["clean"].to(device)
        noisy= data["noisy"].to(device)
        """
        if not hp.model.mag_only : 
            estim =  estim
        else :
            spec= torch.stft(
                noisy,
                n_fft = hp.audio.n_fft,
                hop_length=hp.audio.n_hop,
                window=torch.hann_window(hp.audio.n_fft).to(device)
                )
            spec_real = spec[...,0]
            spec_imag = spec[...,1]
            phase = torch.atan2(spec_real,spec_imag)
            estim_spec = estim[:,0,:,:] * torch.exp(phase*1j)
            estim_wav =  torch.istft(
                estim_spec,
                n_fft = hp.audio.n_fft,
                hop_length=hp.audio.n_hop,
                window=torch.hann_window(hp.audio.n_fft).to(device)
                )
        """
        loss = criterion(estim,noisy.to(device),clean.to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)
    elif hp.loss.type == "mwMSELoss" : 
        loss = criterion(estim,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device)
    elif hp.loss.type== "MSELoss":
        loss = criterion(estim,data["clean"].to(device))
    elif hp.loss.type == "mwMSELoss+wSDRLoss" : 
        estim_wav = torch.istft(estim[:,0,:,:],n_fft = hp.data.n_fft,hop_length=hp.data.n_hop,window=torch.hann_window(hp.data.n_fft).to(device))
        loss = criterion[0](estim,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device) + criterion[1](estim_wav,data["noisy_wav"].to(device),data["clean_wav"].to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)
    else :
        loss = criterion(estim,data["clean"].to(device))

    if loss.isinf().any() : 
        print("Warning::There is inf in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()

    if loss.isnan().any() : 
        print("Warning::There is nan in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()

    if ret_output :
        return estim, loss
    else : 
        return loss

###### from audio_zen.acoustics.feature
def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)

def MRI(X):
    mag, _ = mag_phase(X)
    mag = mag.unsqueeze(0)
    real = (X.real).unsqueeze(0)
    imag = (X.imag).unsqueeze(0)

    return mag.float(),real.float(),imag.float()

