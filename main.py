from src.config import load_config
from src.train import train
from src.metrics import metrics


def main():
    config = load_config("conf.yml")
    print('Config loaded')
    mode = config.MODE
    if mode == 1:
        train(config)
    else:
        print("performing metric calculations")
        metrics(config)

if __name__ == "__main__":
    !pip install wandb -q
    !pip install POT -q
    main()