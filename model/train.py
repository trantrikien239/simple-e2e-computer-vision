# This file defines a trainer class for training the classifier. After 
# training, the model is saved as a safe tensor.
# Use W&B to log the loss and accuracy metrics during training.

import os
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from mnist_cnn import MnistMLP
from mnist_dataset import get_mnist_data_loader
import wandb

@dataclass
class Config:
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 10
    weight_decay: float = 0.01

    def to_dict(self):
        return self.__dict__


class Trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model = MnistMLP().to(self.device)
        self.train_loader, self.test_loader = get_mnist_data_loader(self.config['batch_size'])
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.wandb = wandb
        self.wandb.init(project='mnist-classifier', config=self.config)
        self.wandb.watch(self.model)
        
    def train(self):
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            train_correct = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss /= len(self.train_loader.dataset)
            train_accuracy = train_correct / len(self.train_loader.dataset)
            self.wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy})
            
            self.model.eval()
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = test_correct / len(self.test_loader.dataset)
            self.wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})
            
            print(f'Epoch {epoch + 1}/{self.config["epochs"]}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            
if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config.to_dict())
    print(f'Model: {trainer.model.__class__.__name__}. Size: {trainer.model.count_parameters()}')
    
    current_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f'Begin training on device {trainer.device} at {current_ts}...')
    
    trainer.train()

    save_path = f"model_registry/{current_ts}"
    os.makedirs(save_path, exist_ok=True)

    torch.save(trainer.model.state_dict(), f'{save_path}/model.pth')
    trainer.wandb.save('model.pth')