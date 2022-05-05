import torch
from torch.autograd import Variable
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from jma_pytorch_dataset import *
from utils import AverageMeter, Logger
from criteria_precip import *
# for debug
from tools_mem import *

# training/validation for one epoch

# --------------------------
# Training
# --------------------------

def train_epoch(epoch,num_epochs,train_loader,model,loss_fn,optimizer,train_logger,train_batch_logger,opt,scl):
    
    print('train at epoch {}'.format(epoch))

    losses = AverageMeter()
    
    for i_batch, sample_batched in enumerate(train_loader):
        #print(i_batch, sample_batched['past'].size(),sample_batched['future'].size())
        input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = input # set the input field as the target 
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(input)
        # use opt.tdim_loss steps
        loss = loss_fn(output[:,0:opt.tdim_loss,:,:,:], target[:,0:opt.tdim_loss,:,:,:])
        # loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # for logging
        losses.update(loss.item(), input.size(0))

        print('chk lr ',optimizer.param_groups[0]['lr'])
        train_batch_logger.log({
            'epoch': epoch,
            'batch': i_batch+1,
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        if (i_batch+1) % 1 == 0:
            print ('Train Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch, num_epochs, i_batch+1, len(train_loader.dataset)//train_loader.batch_size, loss.item()))

    # update lr for optimizer
    optimizer.step()

    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    # free gpu memory
    del input,target,output,loss

# --------------------------
# Validation
# --------------------------

def valid_epoch(epoch,num_epochs,valid_loader,model,loss_fn,valid_logger,opt,scl):
    print('validation at epoch {}'.format(epoch))
    
    losses = AverageMeter()
        
    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(valid_loader):
        input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        
        # Forward
        output = model(input)
        loss = loss_fn(output, target)

        # for logging
        losses.update(loss.item(), input.size(0))

        if (i_batch+1) % 1 == 0:
            print ('Valid Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch, num_epochs, i_batch+1, len(valid_loader.dataset)//valid_loader.batch_size, loss.item()))
            
    valid_logger.log({
        'epoch': epoch,
        'loss': losses.avg})
    # free gpu memory
    del input,target,output,loss

# --------------------------
# Test
# --------------------------

def test_epoch(test_loader,model,loss_fn,opt,scl):
    print('Testing for the model')
    
    # initialize
    MSE_all = np.empty((0,opt.tdim_use),float)

    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(test_loader):
        input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        
        # Forward
        output = model(input)
        loss = loss_fn(output, target)
        
        # apply evaluation metric
        Xtrue = scl.inv(target.data.cpu().numpy())
        Xmodel = scl.inv(output.data.cpu().numpy())

        if (i_batch+1) % 1 == 0:
            print ('Test Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.item()))
    # save evaluated metric as csv file
    tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
    # import pdb; pdb.set_trace()
    df = pd.DataFrame({'tpred_min':tpred,
                       'MSE': MSE})
    fname = 'test_evaluation_predtime.csv' % (opt.test_tail)
    df.to_csv(os.path.join(opt.result_path,fname), float_format='%.3f')
    # free gpu memory
    del input,target,output,loss
    
