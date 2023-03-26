import os
from functools import partial

from network import UNetModel, EMA
from dataloader import collate_fn_emd, collate_fn_sinkhorn, ShoeBagDataset, ColoredDataset, load_dataset, load_shoes_bags
from diffusion import GaussianDiffusion,extract

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
import wandb

from plotters import print_stat, fig2data, fig2img, plot_noise, plot_y

import numpy as np
from tqdm import tqdm
import copy

class Trainer():
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.diffusion = GaussianDiffusion(config.IMAGE_SIZE,config.CHANNEL_X,config.CHANNEL_Y,config.TIMESTEPS)
        in_channels = config.CHANNEL_X + config.CHANNEL_Y
        out_channels = config.CHANNEL_Y
        self.network = UNetModel(
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
            ).to(self.device)
        
        if config.DATA == "mnist":

            train_set2, test_set2 = load_dataset(name="MNIST-colored_2", path="")
            train_set3, test_set3 = load_dataset(name="MNIST-colored_3", path="")
            
            train_dataset = ColoredDataset(train_set2, train_set3)
            test_dataset = ColoredDataset(test_set2, test_set3)

        elif config.DATA == "shoes-bags":

            !FILEID='1i1F462P45I2w3lIFL-u8gwmcPm3iEAGb' && \
            FILENAME='shoes_bags_data.zip' && \
            FILEDEST="https://docs.google.com/uc?export=download&id=${FILEID}" && \
            wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${FILEDEST} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

            !unzip shoes_bags_data.zip

            train_set_bags, test_set_bags = load_shoes_bags(data_path="./shoes_tensor_32.torch")
            train_set_shoes, test_set_shoes = load_shoes_bags(data_path="./handbag_tensor_32.torch")
            
            train_dataset = ShoeBagDataset(train_set_bags, train_set_shoes)
            test_dataset = ShoeBagDataset(test_set_bags, test_set_shoes)

            
        self.batch_size = config.BATCH_SIZE
        self.batch_size_val = config.BATCH_SIZE_VAL
        
        self.dataloader_train = torch.utils.data.DataLoader(
                                    train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=collate_fn_sinkhorn if config.REG else collate_fn_emd,
                                    drop_last=True
                                )

        self.dataloader_validation = torch.utils.data.DataLoader(
                                    test_dataset,
                                    batch_size=self.batch_size_val,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn_sinkhorn if config.REG else collate_fn_emd,
                                    drop_last=True
                                )
        
        self.iteration_max = config.ITERATION_MAX
        self.EMA = EMA(0.9999)
        self.LR = config.LR
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        if config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else :
            print('Loss not implemented, setting the loss to L2 (default one)')
        self.num_timesteps = config.TIMESTEPS
        self.validation_every = config.VALIDATION_EVERY
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.ema_model = copy.deepcopy(self.network).to(self.device)
        self.plot_every = config.PLOT_EVERY
    def save_model(self,name,EMA=False):
        if not EMA:
            torch.save(self.network.state_dict(),name)
        else:
            torch.save(self.ema_model.state_dict(),name)

    def train(self):

        to_torch = partial(torch.tensor, dtype=torch.float32)
        optimizer = optim.Adam(self.network.parameters(),lr=self.LR)
        iteration = 0
        
        print('Starting Training')
        
        wandb_step = 0

        while iteration < self.iteration_max:

            print(f"Start of interation no. {iteration+1}")

            tq = tqdm(self.dataloader_train)
            
            for step, (grey, color) in enumerate(tq):
                tq.set_description(f'Iteration {iteration} / {self.iteration_max}')
                self.network.train()
                optimizer.zero_grad()

                t = torch.randint(0, self.num_timesteps, (self.batch_size,)).long()
                
                noisy_image,noise_ref = self.diffusion.noisy_image(t,color)
                noise_pred = self.diffusion.noise_prediction(self.network,noisy_image.to(self.device),grey.to(self.device),t.to(self.device))
                loss = self.loss(noise_ref.to(self.device),noise_pred)
                loss.backward()
                optimizer.step()
                tq.set_postfix(loss = loss.item())
                
                wandb.log({f'Noise_loss_train' : loss.item()}, step=wandb_step)
                
                iteration+=1

                if iteration%self.ema_every == 0 and iteration>self.start_ema:
                    print('EMA update')
                    self.EMA.update_model_average(self.ema_model,self.network)

                if iteration%self.plot_every == 0:
                    fig, ax = plot_noise(noise_ref.cpu().detach(), noise_pred.cpu().detach())
                    wandb.log({'Noise': [wandb.Image(fig2img(fig))]}, step=wandb_step)

                if iteration%self.save_model_every == 0:
                    print('Saving models')
                    if not os.path.exists('models/'):
                        os.makedirs('models')
                    self.save_model(f'models/model_{iteration}.pth')
                    self.save_model(f'models/model_ema_{iteration}.pth',EMA=True)
                    
                wandb_step += 1

                if iteration%self.validation_every == 0:
                    tq_val = tqdm(self.dataloader_validation)
                    with torch.no_grad():
                        self.network.eval()
                        
                        y_mean_norms = []
                        for val_step, (grey, color) in enumerate(tq_val):
                            tq_val.set_description(f'Iteration {iteration} / {self.iteration_max}')
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
                

                                time = (torch.ones((self.batch_size_val,)) * t).long()
                                y = extract(to_torch(np.sqrt(1/alphas)),time,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),time,y.shape))*self.network(y.to(self.device),grey.to(self.device),time.to(self.device)).detach().cpu()) + extract(to_torch(np.sqrt(1-alphas)),time,z.shape)*z
                                y_ema = extract(to_torch(np.sqrt(1/alphas)),time,y.shape)*(y-(extract(to_torch((1-alphas)/np.sqrt(1-gammas)),time,y.shape))*self.ema_model(y.to(self.device),grey.to(self.device),time.to(self.device)).detach().cpu()) + extract(to_torch(np.sqrt(1-alphas)),time,z.shape)*z

                                y_mean_norms.append(torch.norm(y.reshape(y.shape[0], -1), dim=-1, p=2).mean().cpu().item())
            
                            data = [[x_coord, y_coord] for (x_coord, y_coord) in zip(range(T+1), y_mean_norms)]
                            table = wandb.Table(data=data, columns = ["x", "y"])
                            wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "x", "y",
                                        title="Custom Y vs X Line Plot")})
            
                            
                            fig, ax = plot_y(grey.cpu().detach(), y.cpu().detach(), y_ema.cpu().detach(), color.cpu().detach())
                            wandb.log({'Ys': [wandb.Image(fig2img(fig))]}, step=wandb_step)
                                

                            loss = self.loss(color,y)
                            loss_ema = self.loss(color,y_ema)
                            tq_val.set_postfix({'loss': loss.item(),'loss ema': loss_ema.item()})

                            wandb.log({f'y_loss_valid': loss.item()}, step=wandb_step) 
                            wandb.log({f'ema_loss_valid': loss_ema.item()}, step=wandb_step)
                            
                            #because validation takes too long, we simply use 1 batch
                            break