import argparse, yaml, torch
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.train.trainer import Trainer
from src.model.my_model import MyModel
from src.dataloader.dataset import MyDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', type=str)
    return parser.parse_args()

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.batch_size: config['train']['batch_size'] = args.batch_size
    if args.learning_rate: config['train']['learning_rate'] = args.learning_rate
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.device: config['train']['device'] = args.device
    return config

def main():
    args = parse_args()
    config = load_config(args)
    set_seed(config['train']['seed'])
    device = torch.device(config['train']['device'])

    train_ds = MyDataset('train')
    val_ds = MyDataset('val')
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'])

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    loss_fn = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, config, device)
    trainer.train()

if __name__ == "__main__":
    main()
