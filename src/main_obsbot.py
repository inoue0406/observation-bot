import numpy as np
import torch 
import torchvision

import pandas as pd
import h5py
import os
import json
import time

import mlflow
import mlflow.pytorch

from scaler import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *    

from models.model_obsbot import obsbot, obsbot_observer

device = torch.device("cuda")

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
if __name__ == '__main__':
   
    # parse command-line options
    opt = parse_opts()
    print(json.dumps(vars(opt), indent=2))
    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        opt_file.write(json.dumps(vars(opt), indent=2))

    # Tracking by MLFlow
    experiment_id = mlflow.tracking.MlflowClient().get_experiment_by_name(opt.result_path[0:10])
    
    # generic log file  
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # model information
    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')

    # prepare scaler for data
    if opt.dataset == 'radarJMA':
        if opt.data_scaling == 'linear':
            scl = LinearScaler()
    elif opt.dataset == 'artfield':
        if opt.data_scaling == 'linear':
            # use identity transformation, since the data is already scaled
            scl = LinearScaler(rmax=1.0)

    # initialize Observer/Policy/Predictor instance
    if opt.model_name == 'obsbot':

        model = obsbot(opt.image_size,
                       opt.pc_size, 
                       opt.batch_size,
                       opt.model_mode,
                       opt.observer_type,
                       opt.policy_type,
                       opt.predictor_type,
                       opt.freeze,
                       opt.observer_transfer_path,
                       opt.interp_type,
                       opt.pc_initialize).to(device)
                       
    elif opt.model_name == 'observer':
        # Observer-only Model

        model = obsbot_observer(opt.image_size,
                       opt.pc_size, 
                       opt.batch_size,
                       opt.model_mode,
                       opt.observer_type,
                       opt.interp_type,
                       opt.pc_initialize).to(device)
        
    # Data Parallel Multi-GPU Run
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model) # make parallel
    model.to(device)
    
    if not opt.no_train:
        # loading datasets
        if opt.dataset == 'radarJMA':
            from jma_pytorch_dataset import *
            train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                            csv_file=opt.train_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
            
            valid_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.valid_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
        elif opt.dataset == 'artfield':
            from artfield_pytorch_dataset import *
            train_dataset = ArtfieldDataset(root_dir=opt.data_path,
                                            csv_file=opt.train_path,
                                            mode=opt.model_mode,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
            
            valid_dataset = ArtfieldDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.valid_path,
                                            mode=opt.model_mode,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=True)
#                                                   shuffle=False)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=False)
            
        modelinfo.write('Model Structure \n')
        modelinfo.write(str(model))
        count_parameters(model,modelinfo)
        modelinfo.close()
        
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()

        # Type of optimizers adam/rmsprop
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate)
            
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
            
        # Prep logger
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'loss', 'lr'])
        valid_logger = Logger(
            os.path.join(opt.result_path, 'valid.log'),
            ['epoch', 'loss'])
    
        # training 
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(vars(opt))

            for epoch in range(1,opt.n_epochs+1):
                # training & validation
                if opt.model_name == 'obsbot':
                    train_epoch(epoch,opt.n_epochs,train_loader,model,loss_fn,optimizer,
                                train_logger,train_batch_logger,opt,scl)
                    #valid_epoch(epoch,opt.n_epochs,valid_loader,model,loss_fn,
                    #            valid_logger,opt,scl)
                elif opt.model_name == 'observer':
                    train_epoch_observer(epoch,opt.n_epochs,train_loader,model,loss_fn,optimizer,
                                        train_logger,train_batch_logger,opt,scl)
                # step scheduler
                scheduler.step()

                # log time with mlflow
                total_time = time.time() - tstart
                mlflow.log_metric("Elapsed Time", total_time, step=epoch)

                if epoch % opt.checkpoint == 0:
                    # save the trained model for every checkpoint
                    # (1) as binary 
                    #torch.save(model,os.path.join(opt.result_path,
                    #                                 'trained_obsbot_epoch%03d.model' % epoch))
                    # (2) as state dictionary
                    torch.save(model.state_dict(),
                            os.path.join(opt.result_path,
                                            'trained_obsbot_epoch%03d.dict' % epoch))
            # save the trained model
            # (1) as binary 
            #torch.save(model,os.path.join(opt.result_path, 'trained_obsbot.model'))
            # (2) as state dictionary
            if opt.model_name == 'obsbot':
                model_fname = 'trained_obsbot.dict'
                torch.save(model.state_dict(),
                           os.path.join(opt.result_path, model_fname))
            elif  opt.model_name == 'observer':
                model_fname = 'trained_observer.dict'
                torch.save(model.observer.state_dict(),
                           os.path.join(opt.result_path, model_fname))
            mlflow.pytorch.log_model(model, artifact_path="Final")

    # test datasets if specified
    batch_size_test = 4
    if opt.test:
        if opt.no_train:
            #load pretrained model from results directory
            model_fname = os.path.join(opt.result_path, 'trained_obsbot.dict')
            print('loading pretrained model:',model_fname)
            # ###if the model is pickle
            #model_ld = torch.load(model_fname)
            # ###if the model is state dict
            model.load_state_dict(torch.load(model_fname))
            # tweak
            #from models_trajGRU.model_euler_lagrange import EF_el
            #model = EF_el(model.encoder, model.forecaster,
            #              opt.image_size, opt.pc_size, batch_size_test, opt.model_mode, opt.interp_type).to(device)
            #del model_ld
            loss_fn = torch.nn.MSELoss()

        # smaller batch size is used, since trajGRU is heavy on memory
        #batch_size_test = 4
        #batch_size_test = opt.batch_size
        # prepare loader
        if opt.dataset == 'radarJMA':
            from jma_pytorch_dataset import *
            test_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.test_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
                        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size_test,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=False)
        
        # testing for the trained model
        test_epoch(test_loader,model,loss_fn,opt,scl)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
