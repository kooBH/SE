import argparse
import os
import glob
import numpy as np
import librosa as rs
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from Dataset.DatasetDNS import DatasetDNS
from utils.hparams import HParam
from utils.writer import MyWriter
from utils.metric import run_metric
from ptflops import get_model_complexity_info
from common import run,get_model, evaluate_lsnr, set_seed

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

    hp = HParam(args.config,args.default,merge_except=["architecture"])
    print("NOTE::Loading configuration {} based on {}".format(args.config,args.default))

    device = args.device
    torch.cuda.set_device(device)
    version = args.version_name
    num_workers = hp.train.num_workers

    batch_size = hp.train.batch_size
    num_epochs = args.epoch
    print("num_epochs : {}".format(num_epochs))

    best_loss = 1e7
    best_metric = {}
    for m in hp.log.eval : 
        best_metric[m] = 0.0

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    ## Loss
    from utils.Loss import ListLoss, LSNRLoss
    criterion = ListLoss(
        hp.loss,
        hp.loss.ListLoss.list,
        hp.loss.ListLoss.weight,
        )
    criterion_lsnr = LSNRLoss()

    ##  Dataset
    train_dataset = DatasetDNS(hp,is_train=True)
    test_dataset  = DatasetDNS(hp,is_train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)

    model = get_model(hp,device=device)
    #macs_ptflos, params_ptflops = get_model_complexity_info(model, (16000), as_strings=False,print_per_layer_stat=False,verbose=False)   
    #print("ptflops : MACS {} |  PARAM {}".format(macs_ptflos,params_ptflops))

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        try : 
            model.load_state_dict(torch.load(args.chkpt, map_location=device)["model"])
        except KeyError :
            model.load_state_dict(torch.load(args.chkpt, map_location=device))
    print("Model Initialized.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.train.AdamW.lr)

    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min) 
    else :
        raise Exception("Unsupported sceduler type : {}".format(hp.scheduler.type))
    step = args.step
    cnt_log = 0

    ## Eval data load
    list_eval = []
    for path_noisy in glob.glob(os.path.join(hp.data.eval.noisy,"*.wav")) : 
        basename = os.path.basename(path_noisy)
        path_clean = os.path.join(hp.data.eval.clean,basename)
        list_eval.append([path_noisy,path_clean])

    list_DNS_noisy = glob.glob(os.path.join(hp.data.eval.DNS,"noisy","*.wav"),recursive=True)

    list_DNS=[]
    for path_noisy in list_DNS_noisy :
        token = path_noisy.split("/")[-1]
        token = token.split("_")
        fileid = token[-1].split(".")[0]
        path_clean = os.path.join(hp.data.eval.DNS,"clean","clean_fileid_{}.wav".format(fileid))
        list_DNS.append((path_noisy,path_clean))

    list_VD_SNR_noisy = glob.glob(os.path.join(hp.data.eval.VD_SNRm10,"noisy","*.wav"))
    list_VD_SNRm10 = []
    for path_noisy in list_VD_SNR_noisy:
        token = path_noisy.split("/")[-1]
        fileid = token.split(".")[0]
        path_clean = os.path.join(hp.data.eval.VD_SNRm10,"clean","{}.wav".format(fileid))
        list_VD_SNRm10.append((path_noisy,path_clean))

    list_VD_SNR_noisy = glob.glob(os.path.join(hp.data.eval.VD_SNRm10,"noisy","*.wav"))
    list_VD_SNRm5 = []
    for path_noisy in list_VD_SNR_noisy :
        token = path_noisy.split("/")[-1]
        fileid = token.split(".")[0]
        path_clean = os.path.join(hp.data.eval.VD_SNRm5,"clean","{}.wav".format(fileid))
        list_VD_SNRm5.append((path_noisy,path_clean))

    list_VD_SNR_noisy = glob.glob(os.path.join(hp.data.eval.VD_SNR0,"noisy","*.wav"))
    list_VD_SNR0 = []
    for path_noisy in list_VD_SNR_noisy :
        token = path_noisy.split("/")[-1]
        fileid = token.split(".")[0]
        path_clean = os.path.join(hp.data.eval.VD_SNR0,"clean","{}.wav".format(fileid))
        list_VD_SNR0.append((path_noisy,path_clean))

    list_VD_SNR_noisy = glob.glob(os.path.join(hp.data.eval.VD_SNR0,"noisy","*.wav"))
    list_VD_SNRp5 = []
    for path_noisy in list_VD_SNR_noisy :
        token = path_noisy.split("/")[-1]
        fileid = token.split(".")[0]
        path_clean = os.path.join(hp.data.eval.VD_SNRp5,"clean","{}.wav".format(fileid))
        list_VD_SNRp5.append((path_noisy,path_clean))

    list_VD_SNR_noisy = glob.glob(os.path.join(hp.data.eval.VD_SNRp5,"noisy","*.wav"))
    list_VD_SNRp10 = []
    for path_noisy in list_VD_SNR_noisy :
        token = path_noisy.split("/")[-1]
        fileid = token.split(".")[0]
        path_clean = os.path.join(hp.data.eval.VD_SNRp10,"clean","{}.wav".format(fileid))
        list_VD_SNRp10.append((path_noisy,path_clean))

    print("train: {}".format(len(train_dataset)))
    print("test : {}".format(len(test_dataset)))

    writer = MyWriter(log_dir)

    ### TRAIN ####
    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        log_loss = 0
        log_loss_lsnr = 0
        for i, data in enumerate(train_loader):
            step +=data[list(data.keys())[0]].shape[0]

            feature = data["noisy"].to(device)
            estim, lsnr = model(feature)
            loss = criterion(estim,data["clean"].to(device))
            loss_lsnr = criterion_lsnr(lsnr,data["clean"].to(device), data["noisy"].to(device))

            if loss_lsnr.isnan().any() or loss_lsnr.isinf().any():
                import pdb
                pdb.set_trace()

            optimizer.zero_grad()

            if type(loss) is not list : 
                loss.backward(retain_graph=True)
            else :
                for l in loss : 
                    l.backward(retain_graph=True)

            loss_lsnr.backward()

            optimizer.step()

            train_loss += loss.item()
            log_loss += loss.item()
            log_loss_lsnr += loss_lsnr.item()


            if cnt_log //  hp.train.summary_interval > 0:
                log_loss /= max(cnt_log,1)
                log_loss_lsnr /= max(cnt_log,1)
                
                
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4e} Loss[LSNR] : {:.4e}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), log_loss,log_loss_lsnr))
                writer.log_value(log_loss,step,'train loss : '+hp.loss.type)
                writer.log_value(log_loss_lsnr,step,'train loss lsnr')

                log_loss = 0
                cnt_log =  0
            if device == "cuda:1":
                time.sleep(0.01)
            cnt_log += batch_size

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        cnt_eval = 0
        model.eval()
        with torch.no_grad():
            test_loss =0.
            test_loss_lsnr = 0.
            for j, (data) in enumerate(test_loader):
                feature = data["noisy"].to(device)
                estim, lsnr = model(feature)
                loss = criterion(estim,data["clean"].to(device))
                loss_lsnr = criterion_lsnr(lsnr,data["clean"].to(device), data["noisy"].to(device))

                if loss_lsnr.isnan().any() or loss_lsnr.isinf().any():
                    import pdb
                    pdb.set_trace()

                if loss is None : 
                    continue
                test_loss += loss.item()
                test_loss_lsnr += loss_lsnr.item()

            test_loss = test_loss/len(test_loader)
            test_loss_lsnr = test_loss_lsnr/len(test_loader)

            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Loss[LSNR] : {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), test_loss, test_loss_lsnr))

            if hp.scheduler.type == 'Plateau' :
                scheduler.step(test_loss)
            elif hp.scheduler.type == 'oneCycle' :
                scheduler.step()
            elif hp.scheduler.type == "CosineAnnealingLR" :
                scheduler.step()
            elif hp.scheduler.type == "StepLR" :
                scheduler.step()

            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)
            writer.log_value(test_loss_lsnr,step,'test loss lsnr')

            writer.log_spec(data["noisy"][0],"noisy_s",step)
            writer.log_spec(estim[0],"estim_s",step)
            writer.log_spec(data["clean"][0],"clean_s",step)

            writer.log_audio(data["noisy"][0],"noisy_a",step)
            writer.log_audio(estim[0],"estim_a",step)
            writer.log_audio(data["clean"][0],"clean_a",step)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

            if epoch % hp.train.eval_interval == 0 or epoch == num_epochs-1:
                ## Metric
                metric = evaluate_lsnr(hp,model,list_eval,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_VD")

                metric = evaluate_lsnr(hp,model,list_VD_SNRm10,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_VD_SNRm10")

                metric = evaluate_lsnr(hp,model,list_VD_SNRm5,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_VD_SNRm5")

                metric = evaluate_lsnr(hp,model,list_VD_SNR0,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_VD_SNR0")

                metric = evaluate_lsnr(hp,model,list_VD_SNRp5,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_VD_SNRp5")

                metric = evaluate_lsnr(hp,model,list_VD_SNRp10,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_VD_SNRp10")

                metric_dns = evaluate_lsnr(hp,model,list_DNS,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric_dns[m],step,m+"_DNS")

                    if metric_dns[m] > best_metric[m] : 
                        best_metric[m] = metric_dns[m]
                        torch.save(model.state_dict(), str(modelsave_path)+f"/best_{m}.pt")

    writer.close()