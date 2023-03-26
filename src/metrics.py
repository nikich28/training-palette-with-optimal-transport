import os
from functools import partial

from network import UNetModel, EMA
from dataloader import collate_fn_emd, collate_fn_sinkhorn, ShoeBagDataset, ColoredDataset, load_dataset, load_shoes_bags
from diffusion import GaussianDiffusion,extract

from ignite.metrics import SSIM, PSNR

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
import wandb

from plotters import print_stat, fig2data, fig2img, plot_noise, plot_y

import numpy as np
from tqdm import tqdm
import copy

#for metrics
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *


def eval_step(engine, batch):
    return batch


def metrics(config):
    default_evaluator = Engine(eval_step)


    #get checkpoints using kaggle, for example

    !kaggle kernels output kaggle3223/notebook04a4ac77c4 -p ./

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion = GaussianDiffusion(config.IMAGE_SIZE,config.CHANNEL_X,config.CHANNEL_Y,config.TIMESTEPS)
    in_channels = config.CHANNEL_X + config.CHANNEL_Y
    out_channels = config.CHANNEL_Y
    network = UNetModel(
        config.IMAGE_SIZE,
        in_channels,
        config.MODEL_CHANNELS,
        out_channels,
        config.NUM_RESBLOCKS,
        config.ATTENTION_RESOLUTIONS,
        config.DROPOUT,
        config.CHANNEL_MULT,
        config.CONV_RESAMPLE,
        config.USE_CHECKPOINT,
        config.USE_FP16,
        config.NUM_HEADS,
        config.NUM_HEAD_CHANNELS,
        config.NUM_HEAD_UPSAMPLE,
        config.USE_SCALE_SHIFT_NORM,
        config.RESBLOCK_UPDOWN,
        config.USE_NEW_ATTENTION_ORDER,
        ).to(device)


    to_torch = partial(torch.tensor, dtype=torch.float32)
    batch_size_val = 64

    if config.DATA == "mnist":

        _, test_set2 = load_dataset(name="MNIST-colored_2", path="")
        _, test_set3 = load_dataset(name="MNIST-colored_3", path="")

        test_dataset = ColoredDataset(test_set2, test_set3)
    
    elif config.DATA == "shoes-bags":

        !FILEID='1i1F462P45I2w3lIFL-u8gwmcPm3iEAGb' && \
        FILENAME='shoes_bags_data.zip' && \
        FILEDEST="https://docs.google.com/uc?export=download&id=${FILEID}" && \
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${FILEDEST} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

        !unzip shoes_bags_data.zip

        _, test_set_bags = load_shoes_bags(data_path="./shoes_tensor_32.torch")
        _, test_set_shoes = load_shoes_bags(data_path="./handbag_tensor_32.torch")

        test_dataset = ShoeBagDataset(test_set_bags, test_set_shoes)


    dataloader_validation = torch.utils.data.DataLoader(
                                test_dataset,
                                batch_size=batch_size_val,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=collate_fn_sinkhorn if config.REG else collate_fn_emd,
                                drop_last=True
                            )

    LR = config.LR

    num_timesteps = config.TIMESTEPS

    ema_model = copy.deepcopy(network).to(device)

    network.load_state_dict(torch.load("models/model_best.pth")["model"])
    ema_model.load_state_dict(torch.load("models/model_ema_best.pth")["model"])

    tq_val = tqdm(dataloader_validation)

    with torch.no_grad():
        network.eval()
        
        psnrs = []
        psnrs_ema = []
        ssims = []
        ssims_ema = []
        
        i = 0
        for val_step, (grey, color) in enumerate(tq_val):

            T = 1000

            betas = np.linspace(1e-6,0.01,T)
            alphas = 1. - betas
            gammas = to_torch(np.cumprod(alphas,axis=0))
            alphas = to_torch(alphas)
            betas = to_torch(betas)

            y = torch.randn_like(color)
            y_norm = []
            y_norm_ema = []
            for t in reversed(range(T)):
                if t == 0 :
                    z = torch.zeros_like(color)
                else:
                    z = torch.randn_like(color)


                time = (torch.ones((batch_size_val,)) * t).long()
                y = extract(to_torch(np.sqrt(1/alphas)),time,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),time,y.shape))*network(y.to(device),grey.to(device),time.to(device)).detach().cpu()) + extract(to_torch(np.sqrt(1-alphas)),time,z.shape)*z
                y_ema = extract(to_torch(np.sqrt(1/alphas)),time,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),time,y.shape))*ema_model(y.to(device),grey.to(device),time.to(device)).detach().cpu()) + extract(to_torch(np.sqrt(1-alphas)),time,z.shape)*z
                
                
            psnr = PSNR(data_range=1.0)
            psnr.attach(default_evaluator, 'psnr')
            state = default_evaluator.run([[y.mul(0.5).add(0.5), color.mul(0.5).add(0.5)]])
            psnrs.append(state.metrics['psnr'])
            print("PSNR: ", psnrs[-1])
            
            state = default_evaluator.run([[y_ema.mul(0.5).add(0.5), color.mul(0.5).add(0.5)]])
            psnrs_ema.append(state.metrics['psnr'])
            print("PSNR ema: ", psnrs_ema[-1])
            
            metric = SSIM(data_range=1.0)
            metric.attach(default_evaluator, 'ssim')
            state = default_evaluator.run([[y.mul(0.5).add(0.5), color.mul(0.5).add(0.5)]])
            ssims.append(state.metrics['ssim'])
            print("SSIM: ", ssims[-1])
            
            state = default_evaluator.run([[y_ema.mul(0.5).add(0.5), color.mul(0.5).add(0.5)]])
            ssims_ema.append(state.metrics['ssim'])
            print("SSIM ema: ", ssims_ema[-1])
            
            
            i += 1
            if i == 15:
                break
                
    print("Total PSNR: ", np.mean(psnrs))
    print("Total PSNR ema: ", np.mean(psnrs_ema))
    print("Total SSIM: ", np.mean(ssims))
    print("Total SSIM ema: ", np.mean(ssims_ema))