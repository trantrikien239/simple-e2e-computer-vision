# This file defines a trainer class for training the classifier. After 
# training, the model is saved as a safe tensor.
# Use W&B to log the loss and accuracy metrics during training.

import os
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import v2

from mnist_cnn import MnistMLP
from mnist_dataset import get_mnist_data_loader, \
    get_data_loader_from_image_folder, GaussianNoise
import wandb

@dataclass
class Config:
    batch_size: int = 512
    learning_rate: float = 0.001
    epochs: int = 10
    weight_decay: float = 0.01
    augmentations: list = None

    def to_dict(self):
        return self.__dict__
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:
    def __init__(self, config, eval_sets=None):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model = MnistMLP().to(self.device)
        self.train_loader, self.test_loader = get_mnist_data_loader(
            self.config['batch_size'], augmentations=self.config['augmentations'])
        if eval_sets is None:
            self.eval_sets = {'test_normal_all': self.test_loader}
        else:
            self.eval_sets = eval_sets
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.wandb = wandb
        self.wandb.init(project='mnist-classifier-p2', config=self.config)
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
            print(f'Epoch {epoch + 1}/{self.config["epochs"]}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, ')
            
            self.evaluate()
            
    def evaluate(self):
        self.model.eval()

        for name, eval_set in self.eval_sets.items():
            eval_loss = 0
            eval_correct = 0
            with torch.no_grad():
                for data, target in eval_set:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    eval_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    eval_correct += pred.eq(target.view_as(pred)).sum().item()
            eval_loss /= len(eval_set.dataset)
            eval_accuracy = eval_correct / len(eval_set.dataset)
            self.wandb.log({f'{name}_loss': eval_loss, f'{name}_accuracy': eval_accuracy})
            print(f'{name} Loss: {eval_loss:.4f}, {name} Accuracy: {eval_accuracy:.4f}')
        
            
if __name__ == '__main__':
    config = Config()
    eval_sets = {
        name: get_data_loader_from_image_folder(
            config.batch_size, f'data/MNIST/images/test-sample-1000-seed-42-folders/{name}')
        for name in ['normal', 'noisy', 'bright']
    }

    config.update(augmentations=[
        transforms.RandomApply(torch.nn.ModuleList([
            GaussianNoise(mean=0, sigma=0.1, clip=True)
        ]), p=0.3)
    ])

    trainer = Trainer(config.to_dict(), eval_sets=eval_sets)
    
    print(f'Model: {trainer.model.__class__.__name__}. Size: {trainer.model.count_parameters()}')
    
    current_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f'Begin training on device {trainer.device} at {current_ts}...')
    
    trainer.train()

    save_path = f"model_registry/{current_ts}"
    save_path_latest = "model_registry/latest"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_latest, exist_ok=True)

    scripted_model = torch.jit.script(trainer.model)
    torch.jit.save(scripted_model, f'{save_path}/scripted_model.pt')
    torch.jit.save(scripted_model, f'{save_path_latest}/scripted_model.pt')
    print(f'Model saved at {save_path}')
    # Save the model to W&B
    trainer.wandb.save(f'{save_path}/scripted_model.pt')