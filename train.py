import torch
import os
import numpy as np
from collections import defaultdict
import hydra

from tensorboardX import SummaryWriter

from utils.hparams import HParam
from utils.writer import MyWriter

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from ptflops import get_model_complexity_info

from common import run,get_model, evaluate, set_seed, listing_VD

def get_criterion(hp):
    from utils.Loss import ListLoss
    criterion = ListLoss(
        hp.loss,
        hp.loss.list,
        hp.loss.weight
    )
    return criterion

def get_optimizer(cfg, model,lr=None):
    if cfg.optimizer.type == 'Adam' :
        cls_optimizer = torch.optim.Adam
    elif cfg.optimizer.type == 'AdamW' :
        cls_optimizer = torch.optim.AdamW
    elif cfg.optimizer.type == "AdamP" : 
        from utils.optimizer import AdamP
        cls_optimizer = AdamP
    else :
        raise Exception("ERROR::Unknown optimizer : {}".format(cfg.optimizer))

    if lr is not None:
        print(f"get_optimizer()::Using overridden learning rate: {lr}")
        optimizer = cls_optimizer(model.parameters(), lr=lr, **cfg.optimizer.params)
    else : 
        optimizer = cls_optimizer(model.parameters(), lr=cfg.optimizer.lr, **cfg.optimizer.params)
    return optimizer

def get_scheduler(cfg,optimizer,train_loader = None) :
    if cfg.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=cfg.scheduler.Plateau.mode,
            factor=cfg.scheduler.Plateau.factor,
            patience=cfg.scheduler.Plateau.patience,
            min_lr=cfg.scheduler.Plateau.min_lr)
    elif cfg.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = cfg.scheduler.oneCycle.max_lr,
                epochs=cfg.train.epoch,
                steps_per_epoch = len(train_loader)
          )
    elif cfg.scheduler.type == "LinearPerEpoch" :
        from utils.schedule import LinearPerEpochScheduler
        scheduler = LinearPerEpochScheduler(optimizer, len(train_loader))
    elif cfg.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.CosineAnnealingLR.T_max, eta_min=cfg.scheduler.CosineAnnealingLR.eta_min) 
    elif cfg.scheduler.type == "StepLR" :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.StepLR.step_size, gamma=cfg.scheduler.StepLR.gamma)
    elif cfg.scheduler.type == "CosineAnnealingWarmRestarts" :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.CosineAnnealingWarmRestarts.T_0, T_mult=cfg.scheduler.CosineAnnealingWarmRestarts.T_mult, eta_min=cfg.scheduler.CosineAnnealingWarmRestarts.eta_min)
    elif cfg.scheduler.type == "Fixed" : 
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else :
        raise Exception("Unsupported sceduler type : {}".format(cfg.scheduler.type))
    
    warmup = None
    if cfg.scheduler.use_warmup : 
        from utils.schedule import WarmUpScheduler
        warmup = WarmUpScheduler(optimizer, len(train_loader))
    return scheduler, warmup

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:

    torch.autograd.set_detect_anomaly(True)

    device = cfg.device
    version = cfg.version
    torch.cuda.set_device(device)

    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.epoch
    num_workers = cfg.train.num_workers
    # lr for overriding 
    lr_override = cfg.get("lr_override", None)

    best_loss = 1e7

    ## load
    modelsave_path = cfg.log.root +'/'+'chkpt' + '/' + version
    log_dir = cfg.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(log_dir)

    if cfg.data.type == "DNS" : 
        # TODO
        raise NotImplementedError("DNS dataset is not implemented yet.")
        from Dataset.DatasetDNS import DatasetDNS
        train_dataset = DatasetDNS(cfg, is_train=True)
        test_dataset= DatasetDNS(cfg, is_train=False)
    elif cfg.data.type == "VD" :
        from data.DatasetVD import DatasetVD
        train_dataset = DatasetVD(cfg, is_train=True)
        test_dataset= DatasetVD(cfg, is_train=False)
    else :
        raise Exception("Unsupported dataset : {}".format(cfg.data.type))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = get_model(cfg).to(device)

    macs_ptflos, params_ptflops = get_model_complexity_info(model, (16000,), as_strings=False,print_per_layer_stat=False,verbose=False)   
    print("ptflops : MACS {}M |  PARAM {}K".format(macs_ptflos/1e6,params_ptflops/1e3))

    if cfg.resume_last :
        last_chkpt_path = os.path.join(modelsave_path, "lastmodel.pt")
        if os.path.exists(last_chkpt_path):
            print(f"NOTE::Resuming training from last checkpoint: {last_chkpt_path}")
            model.load_state_dict(torch.load(last_chkpt_path, map_location=device))
        else:
            print(f"WARNING::No last checkpoint found at {last_chkpt_path}. Starting training from scratch.")

    if cfg.chkpt is not None : 
        print('NOTE::Loading pre-trained model : '+ cfg.chkpt)
        model.load_state_dict(torch.load(cfg.chkpt, map_location=device))

    criterion = get_criterion(cfg).to(device)
    optimizer = get_optimizer(cfg, model,lr=lr_override)
    scheduler, warmup = get_scheduler(cfg,optimizer,train_loader)

    # Eval data Load
    list_eval = {}
    for db in cfg.log.eval.DB : 
        if db == "VD" : 
            list_eval["VD"] = listing_VD(cfg.data.eval.VD)
        else : 
            raise NotImplementedError("Evaluation dataset {} is not implemented".format(db))

    step = int(cfg.get("step", 0))
    log_dev_cnt = 0

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss = defaultdict(float)
        for i, (batch_data) in enumerate(train_loader):
            step +=batch_size
            
            loss,loss_dict = run(batch_data,model,criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss["main"] += loss.item()
            for k in loss_dict.keys() :
                train_loss[k] += loss_dict[k]

            if i % cfg.train.train_interval_iter == cfg.train.train_interval_iter-1 :
                print(f"TRAIN::{version}:Epoch[{epoch+1}/{num_epochs}],Step[{i+1}/{len(train_loader)}]|loss:{train_loss['main']:.2e},",end=" ")
                writer.log_value(train_loss["main"],step,'train/main_loss')
                for k in loss_dict.keys() :
                    writer.log_value(loss_dict[k],step,'train/'+k)
                    print(f'{k}:{loss_dict[k]:.2e}',end=",")
                print()
            break

        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### Validation ####
        model.eval()
        with torch.no_grad():
            test_loss = defaultdict(float)
            for j, (batch_data) in enumerate(test_loader):
                loss,loss_dict = run(batch_data,model,criterion)
                test_loss["main"] += loss.item()
                for k in loss_dict.keys() :
                    test_loss[k] += loss_dict[k]
                log_dev_cnt += batch_size
                break

            for k in test_loss.keys() :
                test_loss[k] /= len(test_loader)
            scheduler.step(test_loss["main"])

            print(f"TEST::{version}:Epoch[{epoch+1}/{num_epochs}]|loss:{test_loss['main']:.2e},",end=" ")
            writer.log_value(test_loss["main"],step,'test/main_loss')
            for k in loss_dict.keys() :
                writer.log_value(test_loss[k],step,'test/'+k)
                print(f'{k}:{test_loss[k]:.2e}',end=",")
            print()

            if best_loss > test_loss["main"]:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss["main"]

            #### Evaluation ####
            for set in list_eval.keys() : 
                metric = evaluate(cfg,model,list_eval[set],device=device)
                for m in cfg.log.eval.metric : 
                    writer.log_value(metric[m],step,f"{set}_{m}")
                    print(f"{set}_{m}: {metric[m]:.2e}")

    writer.close()


if __name__ == "__main__":
    main()
