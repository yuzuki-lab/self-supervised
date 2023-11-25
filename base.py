import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import models
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml
import numpy as np
import os
import timm
import random
import wandb
import warnings

def torch_fix_seed(config):
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main():
    warnings.simplefilter('ignore')

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="seed実験",
        name='shuffle_false_1回目',
        tags=["pretrained"],

        # track hyperparameters and run metadata
        config={
        "architecture": config['model'],
        "dataset": "STL10",
        "epochs": config['epoch'],
        })

    device = 'cuda'
    torch_fix_seed(config)
    
    model = timm.create_model(config['model'], pretrained=True, num_classes=config['num_class']).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if config['optimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    elif config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['0.9'])
    
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])


    
    # traindir = os.path.join(config['data_path'], 'train')
    # valdir = os.path.join(config['data_path'], 'val')
    normalize = transforms.Normalize(mean=config['mean'],std=config['std'])
    
    # train_dataset = datasets.ImageFolder(
    #             traindir,
    #             transforms.Compose([
    #             transforms.RandomResizedCrop(config['image_size']),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize]))

    # val_dataset = datasets.ImageFolder(
    #             valdir,
    #             transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(config['image_size']),
    #             transforms.ToTensor(),
    #             normalize,]))

    # transform = transforms.Compose([transforms.ToTensor(), 
    #                             transforms.Resize(224),
    #                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = datasets.STL10(
        "/home/yishido/DATA",
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                transforms.RandomResizedCrop(config['image_size']),
                                transforms.RandomHorizontalFlip(),
                                normalize]),
        split="train"
    )

    val_dataset = datasets.STL10(
        "/home/yishido/DATA",
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize(256),
                                transforms.CenterCrop(config['image_size']),
                                normalize]),
        split="test"
    )


    train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                num_workers=config['num_workers'],
                pin_memory=True, )
                # sampler=train_loader)

    val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True, )
                # sampler=val_loader)

    print(f"Data loaded: there are {len(train_dataset)} train images.")
    print(f"Data loaded: there are {len(val_dataset)} val images.")

    train_loss_list = [] 
    train_accuracy_list = [] 
    val_loss_list = [] 
    val_accuracy_list = [] 

    print("Starting training !")
    for epoch in range(0,config['epoch']):

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device, config)
        val_loss, val_acc = val(val_loader, model, criterion, epoch, device, config)

        print("--------------------------------------------------------------------------------------------")
        print(f"{epoch}epoch")
        print(f"Trian_Loss: {train_loss:.4f}, Train_Accuracy: {train_acc:.4f}")
        print(f"Test_Loss: {val_loss:.4f}, Test_Accuracy: {val_acc:.4f}")

        wandb.log({"train_accuracy": train_acc,
                   "train_loss": train_loss,
                   "val_accuracy": val_acc,
                   "val_loss" : val_loss,
                   "epoch" : epoch
                   })

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)

        if epoch % config['save_epoch'] == 0:
            save_path = config['save_path']
            save_file = f'{save_path}/checkpoint-{epoch}.pth'
            torch.save(model, save_file)
    
    torch.save(model, f'{save_path}/final.pth')

    wandb.alert(
        title='学習が完了しました。',
        text=f'final_val_acc :{val_acc}',
    )

    wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, device, config):
    model.train()

    train_losses = 0
    train_accuracies = 0
    
    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()       

        y_pred_prob = model(images)
        loss = criterion(y_pred_prob, labels)

        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        y_pred_labels = torch.max(y_pred_prob, 1)[1]
        train_accuracies += torch.sum(y_pred_labels == labels).item() / len(labels)

    epoch_train_loss = train_losses / len(train_loader)
    epoch_train_accuracy = train_accuracies / len(train_loader)

    return epoch_train_loss, epoch_train_accuracy

def val(val_loader, model, criterion, epoch, device, config):
    model.eval()

    test_losses = 0
    test_accuracies = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred_prob = model(images)
            
            loss = criterion(y_pred_prob, labels)
            
            test_losses += loss.item()
            y_pred_labels = torch.max(y_pred_prob, 1)[1]
            test_accuracies += torch.sum(y_pred_labels == labels).item() / len(labels)
    
    epoch_test_loss = test_losses / len(val_loader)
    epoch_test_accuracy = test_accuracies / len(val_loader)    
    
    return epoch_test_loss, epoch_test_accuracy

if __name__ == '__main__':
    main()