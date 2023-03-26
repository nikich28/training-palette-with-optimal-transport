import wandb
from trainer import Trainer

def train(config):
    wandb.init(name=config.EXP_NAME, project='Diffusion_Colorization', config=config)
    trainer = Trainer(config)
    trainer.train()
    print('training complete')
