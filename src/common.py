import torch
import torch.nn

from UNet.UNet import UNet
from UNet.ResUNet import ResUNetOnFreq, ResUNet
from FSN.FullSubNet_Plus import FullSubNet_Plus

def get_model(hp,device):
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
        model = ResUNetOnFreq(c_in=c_in,c_out=c_out,n_fft=hp.audio.n_fft,n_block=5,Softplus_thr=hp.model.Softplus_thr,activation = hp.model.activation).to(device)
    elif hp.model.type == "FullSubNetPlus" : 
        model = FullSubNet_Plus(num_freqs = hp.model.n_freq).to(device)

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

    if criterion is None : 
        return estim

    if hp.loss.type =="wSDRLoss" :
        mag,phase = rs.magphase(data["noisy_wav"])
        estim_wav = torch.istft(estim[:,0,:,:],n_fft = hp.data.n_fft,hop_length=hp.data.n_hop,window=torch.hann_window(hp.data.n_fft).to(device))

        loss = criterion(estim_wav,data["noisy_wav"].to(device),data["clean_wav"].to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)


    elif hp.loss.type == "mwMSELoss" : 
        loss = criterion(estim,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device)
    elif hp.loss.type== "MSELoss":
        loss = criterion(estim,data["clean"].to(device))
    elif hp.loss.type == "mwMSELoss+wSDRLoss" : 
        estim_wav = torch.istft(estim[:,0,:,:],n_fft = hp.data.n_fft,hop_length=hp.data.n_hop,window=torch.hann_window(hp.data.n_fft).to(device))

        loss = criterion[0](estim,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device) + criterion[1](estim_wav,data["noisy_wav"].to(device),data["clean_wav"].to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)

    if loss.isinf().any() : 
        import pdb
        pdb.set_trace()

    if loss.isnan().any() : 
        import pdb
        pdb.set_trace()

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

