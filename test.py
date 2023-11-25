import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import ImageFilter, Image
import random
import os
import numpy as np
import warnings
import wandb
import timm


#simsaimのcodeを参考にする

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
    
def torch_fix_seed(seed):
    # seed = config['seed']
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

    # with open('config.yaml', 'r') as config_file:
    #     config = yaml.safe_load(config_file)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="self-supervised",
        name='simsiam',
        # tags=["pretrained"],

        # track hyperparameters and run metadata
        config={
        "architecture": 'vit_tiny_patch16_224',
        "dataset": "flowers",
        "epochs": 100,
        })

    device = 'cuda'
    torch_fix_seed(32)

    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2048)
    model = SimSiam(base_model).to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.STL10(
        "/home/yishido/DATA",
        download=True,
        transform=TwoCropsTransform(transforms.Compose(augmentation)),
        split="train"
    )

    train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=64,
            shuffle = True, 
            num_workers=8,
            pin_memory=True, )
     

    print(f"Data loaded: there are {len(train_dataset)} train images.")
    # print(f"Data loaded: there are {len(val_dataset)} val images.")

    train_loss_list = [] 
    train_accuracy_list = [] 
    # val_loss_list = [] 
    # val_accuracy_list = [] 

    print("Starting training !")
    for epoch in range(0,100):
        loss = train()
        print("--------------------------------------------------------------------------------------------")
        print(f"{epoch}epoch")
        print(f"Loss: {loss}")

        wandb.log({"Loss": loss,
                   "epoch": epoch})
        

def train():
    a


if __name__ == '__main__':
    main()