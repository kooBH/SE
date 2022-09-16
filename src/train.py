import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from DatasetGender import DatasetGender

from utils.hparams import HParam
from utils.writer import MyWriter
from utils.Loss import wSDRLoss,mwMSELoss

from common import run,get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)
    global device

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    ## target
    train_dataset = DatasetGender(hp.data.root_train,hp)
    test_dataset= DatasetGender(hp.data.root_test,hp)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

    model = get_model(hp).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))
    if hp.loss.type == "wSDRLoss" : 
        criterion = wSDRLoss
    elif hp.loss.type == "mwMSELoss":
        criterion = mwMSELoss
    else :
        raise Exception("ERROR::unknown loss : {}".format(hp.loss.type))

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
                )
    else :
        raise Exception("Unsupported sceduler type")

    step = args.step

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, data in enumerate(train_loader):
            step +=data["input"].shape[0]

            loss = run(hp,data,model,criterion,device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           

            if step %  hp.train.summary_interval == 0:
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (data) in enumerate(test_loader):
                estim_spec, loss = run(hp,data,model,criterion,ret_output=True,device=device)
                test_loss += loss.item()

            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), loss.item()))

            test_loss = test_loss/len(test_loader)
            scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test lost : ' + hp.loss.type)
            writer.log_spec(torch.abs(torch.stft(data["noisy_wav"][0],n_fft=hp.data.n_fft,return_complex=True)),"noisy",step)
            writer.log_spec(torch.abs(estim_spec[0,0]),"estim",step)
            writer.log_spec(torch.abs(torch.stft(data["clean_wav"][0],n_fft=hp.data.n_fft,return_complex=True)),"clean",step)

            clean_wav = data["clean_wav"][0].cpu().detach().numpy()
            noisy_wav = data["noisy_wav"][0].cpu().detach().numpy()

            writer.log_audio(clean_wav,"clean",step,sr=hp.data.sr)
            writer.log_audio(noisy_wav,"noisy",step,sr=hp.data.sr)

            estim_wav = torch.istft(estim_spec[0,0],n_fft = hp.data.n_fft)


            estim_wav = estim_wav/torch.max(torch.abs(estim_wav))
            estim_wav = estim_wav.cpu().detach().numpy()

            writer.log_audio(estim_wav,"estim",step,sr=hp.data.sr)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

