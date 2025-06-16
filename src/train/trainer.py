import torch, os
from tqdm import tqdm
from src.utils.logger import get_logger
class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = get_logger('train', config['path']['log_dir'])
        os.makedirs(config['path']['ckpt_dir'], exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.config['train']['epochs']):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch} - Training"):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc=f"Epoch {epoch} - Validation"):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.model(x), y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_model(self, epoch):
        save_path = os.path.join(self.config['path']['ckpt_dir'], f"best_model_epoch{epoch}.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Saved model: {save_path}")
