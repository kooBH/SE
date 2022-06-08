import torch
import argparse
import numpy as np
import torchaudio
import os
import sys
import glob
from dataset.TestsetModel import TestsetModel
from model.Model import Model
from utils.hparams import HParam

from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()

    ## Parameters 
    hp = HParam(args.config)
    print('NOTE::Loading configuration :: ' + args.config)

    device = hp.gpu
    torch.cuda.set_device(device)

    num_epochs = 1
    batch_size = 1
    test_model = args.model
    win_len = hp.audio.frame

    window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
 
    ## dirs 
    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)

    ## Dataset
    # TODO ?
    test_dataset = TestsetModel(hp.data.root,hp)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=1)

    ## Model
    # TODO
    model = Model().to(device)

    model.load_state_dict(torch.load(test_model,map_location=device))
    model.eval()
    print('NOTE::Loading pre-trained model : ' + test_model)


    with torch.no_grad():
        for i, (data,data_dir,data_name) in enumerate(tqdm(test_loader)):
            # TODO
            data_input = data.to(device)
            data_output = model(data_input)

            auido = torch.istft(data_output,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window,center =True, normalized=False,onesided=True,length=int(length)*hp.audio.shift)

            audio = audio.to('cpu')

            # TODO
            ## Normalize
            max_val = torch.max(torch.abs(audio_me_pe))
            audio_me_pe = audio_me_pe/max_val

            ## Save
            torchaudio.save(output_dir+'/'+str(data_dir[0])+'/'+str(data_name[0])+'.wav',src=audio,sample_rate=hp.audio.samplerate)








   
