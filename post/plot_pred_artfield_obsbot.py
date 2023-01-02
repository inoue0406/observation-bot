#
# Plot Predicted Rainfall Data
#
import torch
import numpy as np

import pandas as pd
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from artfield_pytorch_dataset import *
from scaler import *
from train_valid_epoch import *

# Observation Bot Model
from models.model_obsbot import obsbot

device = torch.device("cuda")

# plot a field
def plot_field(X,title,png_fpath,vmin=0,vmax=1):
    plt.imshow(X,vmin=vmin,vmax=vmax,cmap="GnBu",origin='lower')
    plt.colorbar()
    plt.grid()
    plt.title(title)
    plt.show()
    plt.savefig(png_fpath)
    plt.close()

# plot field and point cloud together
def plot_field_pc(R,x,y,r,title,png_fpath,vmin=0,vmax=1):
    # clip xy in [0,1] range
    x_plt = np.clip(x,0,1)*R.shape[0]
    y_plt = np.clip(y,0,1)*R.shape[1]
    plt.imshow(R,vmin=vmin,vmax=vmax,cmap="GnBu",origin='lower')
    plt.colorbar()
    plt.scatter(x_plt, y_plt, c=r, cmap="GnBu", edgecolors="black")
    # set axes range
    plt.xlim(0, R.shape[0])
    plt.ylim(0, R.shape[1])
    plt.grid()
    plt.title(title)
    plt.show()
    plt.savefig(png_fpath)
    plt.close()

# plot comparison of predicted vs ground truth
def plot_comp_prediction(case):

    # path setting
    data_path = opt["data_path"]
    filelist = opt["test_path"] 
    model_fname = case + '/trained_obsbot.dict'
    pic_path = case + '/png/'
    data_scaling = opt["data_scaling"]

    # model setting
    batch_size = opt["batch_size"]
    tdim_use = opt["tdim_use"]
    img_size = opt["image_size"]
    interp_type = opt["interp_type"]

    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # prepare scaler for data
    if data_scaling == 'linear':
        # use identity scaler
        scl = LinearScaler(rmax=1.0)

    # dataset
    valid_dataset = ArtfieldDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    mode="run",
                                    tdim_use=tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    # load the saved model
    model = torch.load(model_fname)

    model_mode = "check"

    model = obsbot(opt["image_size"],
                   opt["pc_size"], 
                   opt["batch_size"],
                   model_mode,
                   opt["observer_type"],
                   opt["policy_type"],
                   opt["predictor_type"],
                   opt["freeze"],
                   opt["observer_transfer_path"],
                   opt["interp_type"],
                   opt["pc_initialize"]).to(device)

    model.load_state_dict(torch.load(model_fname))

    # evaluation mode
    model.eval()
    #
    for i_batch, sample_batched in enumerate(valid_loader):
        fnames = sample_batched['fnames']
        # apply the trained model to the data
        input = Variable(scl.fwd(sample_batched['past'])).cuda()
        target = Variable(scl.fwd(sample_batched['future'])).cuda()
        #input = Variable(sample_batched['past']).cpu()
        #target = Variable(sample_batched['future']).cpu()
        output,r_pc_out,xy_pc_out = model(input)

        for n,fname in enumerate(fnames):
            # convert to cpu
            pic = target[n,:,0,:,:].cpu()
            pic_tg = scl.inv(pic.data.numpy())
            pic = output[n,:,0,:,:].cpu()
            pic_pred = scl.inv(pic.data.numpy())
            # extract point clout xy and value
            x_pc = xy_pc_out[n,:,0,:].detach().cpu().numpy()
            y_pc = xy_pc_out[n,:,1,:].detach().cpu().numpy()
            r_pc = r_pc_out[n,:,0,:].detach().cpu().numpy()
            # print
            print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
            
            for nt in range(12):
                nt_str = '_dt%02d' % nt
                png_fpath = pic_path+'comp_pred_'+str(n)+'_tg'+nt_str+'.png'
                # plot ground truth
                plot_field(pic_tg[nt,:,:],
                                  fname+' ground truth:'+nt_str,png_fpath)
                # plot UV
                png_fpath = pic_path+'comp_pred_'+str(n)+'_pred'+nt_str+'.png'
                plot_field_pc(pic_pred[nt,:,:],x_pc[nt,:],y_pc[nt,:],r_pc[nt,:],
                                  fname+' model prediction:'+nt_str,png_fpath)
        # free GPU memory (* This seems to be necessary if going without backprop)
        del input,target,output
        return
        
if __name__ == '__main__':

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        print('Usage: python plot_comp_prediction.py CASENAME')
        quit()

    case = argvs[1]

    # Get settings information from opts.json file
    with open(os.path.join(case, 'opts.json'), 'r') as opt_file:
        opt = json.load(opt_file)

    plot_comp_prediction(case)


