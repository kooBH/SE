import torch
import argparse
import torchaudio
import os
import glob
import numpy as np
import librosa as rs
import time

from tensorboardX import SummaryWriter

from Dataset.DatasetGender import DatasetGender
from Dataset.DatasetSPEAR import DatasetSPEAR
from Dataset.DatasetDNS import DatasetDNS

from utils.hparams import HParam
from utils.writer import MyWriter
from utils.Loss import wSDRLoss,mwMSELoss,LevelInvariantNormalizedLoss
from utils.metric import run_metric

from common import run,get_model, evaluate

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
    global device

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    if args.epoch is None : 
        num_epochs = hp.train.epoch
    else :
        num_epochs = args.epoch
    print("num_epochs : {}".format(num_epochs))
    num_workers = hp.train.num_workers

    best_loss = 1e7
    best_pesq = 0.0

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version
    csv_dir = hp.log.root+"/csv/"
    seed_dir = hp.log.root+"/seed/"


    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(csv_dir,exist_ok=True)
    os.makedirs(seed_dir,exist_ok=True)

    with open(seed_dir + "seed_"+version+".txt",'w') as f: 
        f.write(str(torch.seed()))

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
        from utils.Loss import TrunetLoss
        criterion = TrunetLoss(
            #default : 4096, 2048, 1024, 512],[1024, 512, 256]
            frame_size_sdr=hp.loss.TRUNetLoss.frame_size_sdr, frame_size_spec= hp.loss.TRUNetLoss.frame_size_spec,
            overlap = hp.loss.TRUNetLoss.overlap
            )
        req_clean_spec = False
    elif hp.loss.type == "HybridLoss":
        from mpSE.loss import HybridLoss
        criterion = HybridLoss([4096, 2048, 1024, 512],[1024, 512, 256],alpha = hp.loss.HybridLoss.alpha)
        req_clean_spec = False
    elif hp.loss.type == "LevelInvariantNormalizedLoss" : 
        criterion = LevelInvariantNormalizedLoss().to(device)
    elif hp.loss.type == "MultiLoss1" : 
        from utils.Loss import MultiLoss1
        criterion = MultiLoss1(
            frame_size_sdr= hp.loss.MultiLoss1.frame_size_sdr,
            frame_size_spec= hp.loss.MultiLoss1.frame_size_spec,
            frame_size_aw = hp.loss.MultiLoss1.frame_size_aw,
            weight_spec = hp.loss.MultiLoss1.weight_spec,
            weight_sdr = hp.loss.MultiLoss1.weight_sdr,
            weight_aw = hp.loss.MultiLoss1.weight_aw)
    elif hp.loss.type == "MultiLoss2" : 
        from utils.Loss import MultiLoss2
        criterion = MultiLoss2(
            frame_size_sdr= hp.loss.MultiLoss1.frame_size_sdr,
            frame_size_spec= hp.loss.MultiLoss1.frame_size_spec,
            weight_spec = hp.loss.MultiLoss1.weight_spec)
    elif hp.loss.type == "MultiDecibelLoss" : 
        from utils.Loss import MultiDecibelLoss
        criterion = MultiDecibelLoss(
            frame_size_sdr=hp.loss.MultiDecibelLoss.frame_size_sdr,
            weight = hp.loss.MultiDecibelLoss.weight
            )

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
    elif hp.scheduler.type == "LinearPerEpoch" :
        from utils.schedule import LinearPerEpochScheduler
        scheduler = LinearPerEpochScheduler(optimizer, len(train_loader))
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min) 
    elif hp.scheduler.type == "StepLR" :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.scheduler.StepLR.step_size, gamma=hp.scheduler.StepLR.gamma)
    else :
        raise Exception("Unsupported sceduler type : {}".format(hp.scheduler.type))
    
    if hp.scheduler.use_warmup : 
        from utils.schedule import WarmUpScheduler
        warmup = WarmUpScheduler(optimizer, len(train_loader))

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

    ## scaler - may occur issue

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

            if hp.scheduler.use_warmup : 
                if epoch == 0 :
                    warmup.step()

            """
            torch.cuda.amp.autocast() is not good with
            SE task. 
            """
#            with torch.cuda.amp.autocast():
            loss = run(hp,data,model,criterion,device=device)
            if loss is None : 
                print("Warning::zero loss")
                continue
            optimizer.zero_grad()

            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            try : 
                loss.backward()
            except RuntimeError as e:
                #import pdb
                #pdb.set_trace()
                print("RuntimeERror!! : {}".format(e))
                continue

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if hp.scheduler.type == 'LinearPerEpoch' :
                scheduler.step()

            train_loss += loss.item()
            log_loss += loss.item()

            if cnt_log %  hp.train.summary_interval == 0:
                log_loss /= max(cnt_log,1)
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), log_loss))
                writer.log_value(log_loss,step,'train loss : '+hp.loss.type)

                log_loss = 0
                cnt_log =  0
            if device == "cuda:1":
                time.sleep(0.01)
            cnt_log += batch_size

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


            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), loss.item()))

            test_loss = test_loss/len(test_loader)

            if hp.scheduler.type == 'Plateau' :
                scheduler.step(test_loss)
            elif hp.scheduler.type == 'oneCycle' :
                scheduler.step()
            elif hp.scheduler.type == "CosineAnnealingLR" :
                scheduler.step()
            elif hp.scheduler.type == "StepLR" :
                scheduler.step()

            estim,loss= run(hp,data,model,criterion,ret_output=
            True,device=device)

            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            writer.log_spec(data["noisy"][0],"noisy_s",step)
            writer.log_spec(estim[0],"estim_s",step)
            writer.log_spec(data["clean"][0],"clean_s",step)

            writer.log_audio(data["noisy"][0],"noisy_a",step)
            writer.log_audio(estim[0],"estim_a",step)
            writer.log_audio(data["clean"][0],"clean_a",step)

            if hp.loss.type == "MultiLoss1" : 
                for key in criterion.losses.keys() : 
                    writer.log_value(criterion.losses[key],step,key)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

            ## Metric
            metric = evaluate(hp,model,list_eval,device=device)
            for m in hp.log.eval : 
                writer.log_value(metric[m],step,m+"_VD")

            metric_dns = evaluate(hp,model,list_DNS,device=device)
            for m in hp.log.eval : 
                writer.log_value(metric_dns[m],step,m+"_DNS")

            if metric_dns["PESQ_WB"] > best_pesq : 
                best_pesq = metric_dns["PESQ_WB"]
                torch.save(model.state_dict(), str(modelsave_path)+"/best_pesq.pt")
            #torch.save(model.state_dict(), str(modelsave_path)+"/model_PESQWB_{}_epoch_{}.pt".format(metric["PESQ_WB"],epoch))

            #if device == "cuda:1":
             #   time.sleep(0.01)

    writer.close()

    ## Log for best model
    model.load_state_dict(torch.load(str(modelsave_path)+'/bestmodel.pt'))
    metric = evaluate(hp,model,list_eval,device=device)

    with open(str(csv_dir)+"/{}.csv".format(version),'w') as f :
        for m in hp.log.eval : 
            f.write("{},".format(m))
        f.write("\n")
        for m in hp.log.eval : 
            f.write("{},".format(metric[m]))
        f.write("\n")
