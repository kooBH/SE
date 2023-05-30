import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa as rs

from tensorboardX import SummaryWriter

from Dataset.DatasetGender import DatasetGender
from Dataset.DatasetSPEAR import DatasetSPEAR
from Dataset.DatasetDNS import DatasetDNS

from utils.hparams import HParam
from utils.writer import MyWriter
from utils.Loss import wSDRLoss,mwMSELoss,LevelInvariantNormalizedLoss
from utils.metric import run_metric

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
    parser.add_argument('--epoch','-e',type=int,required=False,default=None)
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration {} based on {}".format(args.config,args.default))
    global device

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    if args.epoch is None : 
        num_epochs = hp.train.epoch
    else :
        num_epochs = args.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)


    ## Loss
    req_clean_spec = False
    if hp.loss.type == "wSDRLoss" : 
        criterion = wSDRLoss
    elif hp.loss.type == "mwMSELoss":
        criterion = mwMSELoss
        req_clean_spec = True
    elif hp.loss.type == "MSELoss":
        criterion = torch.nn.MSELoss()
        req_clean_spec = True
    elif hp.loss.type == "mwMSELoss+wSDRLoss" : 
        criterion = [mwMSELoss, wSDRLoss]
        req_clean_spec = True
    elif hp.loss.type == "TRUNetLoss":
        from mpSE.loss import TrunetLoss
        criterion = TrunetLoss([4096, 2048, 1024, 512],[1024, 512, 256])
        req_clean_spec = False
    elif hp.loss.type == "LevelInvariantNormalizedLoss" : 
        criterion = LevelInvariantNormalizedLoss().to(device)

    else :
        raise Exception("ERROR::unknown loss : {}".format(hp.loss.type))

    ##  Dataset
    if hp.task == "Gender" : 
        train_dataset = DatasetGender(hp.data.root_train,hp,sr=hp.data.sr,n_fft=hp.data.n_fft,req_clean_spec=req_clean_spec)
        test_dataset= DatasetGender(hp.data.root_test,hp,sr=hp.data.sr,n_fft=hp.data.n_fft,req_clean_spec=req_clean_spec)
    elif hp.task == "SPEAR" : 
        train_dataset = DatasetSPEAR(hp,is_train=True)
        test_dataset  = DatasetSPEAR(hp,is_train=False)
    elif hp.task == "DNS":
        train_dataset = DatasetDNS(hp,is_train=True)
        test_dataset  = DatasetDNS(hp,is_train=False)
    else :
        raise Exception("ERROR::Unknown task : {}".format(hp.task))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)

    model = get_model(hp,device=device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        try : 
            model.load_state_dict(torch.load(args.chkpt, map_location=device)["model"])
        except KeyError :
            model.load_state_dict(torch.load(args.chkpt, map_location=device))

    if hp.train.optimizer == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.Adam)
    elif hp.train.optimizer == 'AdamW' :
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.train.AdamW.lr)
    else :
        raise Exception("ERROR::Unknown optimizer : {}".format(hp.train.optimizer))

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
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min) 
    else :
        raise Exception("Unsupported sceduler type : {}".format(hp.scheduler.type))

    step = args.step
    cnt_log = 0

    scaler = torch.cuda.amp.GradScaler()
    print("len dataset : {}".format(len(train_dataset)))
    print("len dataset : {}".format(len(test_dataset)))

    writer = MyWriter(log_dir)

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        log_loss = 0
        for i, data in enumerate(train_loader):
            step +=data[list(data.keys())[0]].shape[0]

            """
            torch.cuda.amp.autocast() is not good with
            SE task. 
            """
#            with torch.cuda.amp.autocast():
            loss = run(hp,data,model,criterion,device=device)
            if loss is None : 
                continue
            optimizer.zero_grad()

            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            try : 
                loss.backward()
            except RuntimeError as e:
                import pdb
                pdb.set_trace()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()
            log_loss += loss.item()

            if cnt_log %  hp.train.summary_interval == 0:
                log_loss /= max(cnt_log,1)
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), log_loss))
                writer.log_value(log_loss,step,'train loss : '+hp.loss.type)

                log_loss = 0
                cnt_log =  0
            cnt_log +=1

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (data) in enumerate(test_loader):
                loss = run(hp,data,model,criterion,ret_output=
                False,device=device)
                if loss is None : 
                    continue
                test_loss += loss.item()

            if loss is None : 
                continue

            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), loss.item()))

            test_loss = test_loss/len(test_loader)
            scheduler.step(test_loss)

            estim,loss= run(hp,data,model,criterion,ret_output=
            True,device=device)

            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            writer.log_spec(data["noisy"][0],"noisy_s",step)
            writer.log_spec(estim[0],"estim_s",step)
            writer.log_spec(data["clean"][0],"clean_s",step)

            writer.log_audio(data["noisy"][0],"noisy_a",step)
            writer.log_audio(estim[0],"estim_a",step)
            writer.log_audio(data["clean"][0],"clean_a",step)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

            ## Metric
            metric = {}
            for m in hp.log.eval : 
                metric["{}_with_reverb".format(m)] = 0.0
                metric["{}_no_reverb".format(m)] = 0.0

            for i in range(hp.log.n_eval) : 
                with_reverb,no_reverb = test_dataset.get_eval(i)

                # with_reverb
                noisy_reverb = rs.load(with_reverb[0],sr=hp.data.sr)[0]
                noisy_reverb = torch.unsqueeze(torch.from_numpy(noisy_reverb),0).to(device)
                estim_reverb = model(noisy_reverb).cpu().detach().numpy()

                target_reverb = rs.load(with_reverb[1],sr=hp.data.sr)[0]

                # no_reverb
                noisy_no_reverb = rs.load(no_reverb[0],sr=hp.data.sr)[0]
                noisy_no_reverb = torch.unsqueeze(torch.from_numpy(noisy_no_reverb),0).to(device)
                estim_no_reverb = model(noisy_no_reverb).cpu().detach().numpy()

                target_no_reverb = rs.load(no_reverb[1],sr=hp.data.sr)[0]

                for m in hp.log.eval : 
                    val_reverb = run_metric(estim_reverb[0],target_reverb,m) 
                    metric["{}_with_reverb".format(m)] += val_reverb
                    val_no_reverb = run_metric(estim_no_reverb[0],target_no_reverb,m) 
                    metric["{}_no_reverb".format(m)] += val_no_reverb
                
            for m in hp.log.eval : 
                key = "{}_no_reverb".format(m)
                metric[key] /= hp.log.n_eval
                writer.log_value(metric[key],step,key)

                key = "{}_with_reverb".format(m)
                metric[key] /= hp.log.n_eval
                writer.log_value(metric[key],step,key)
    writer.close()
