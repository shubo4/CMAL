import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import argparse

parser = argparse.ArgumentParser(description='Required inputs')
parser.add_argument('--data_path',  type=str, help='path to data directory')  # data path contains train and test folder with images
parser.add_argument('--epochs',     type=int, default = 50,        help='epochs to run for')  
parser.add_argument('--batch_size', type=int, default= 32,         help='batch size')        
parser.add_argument('--output_path',type=str, default= './outputs',help='output directory')  #output directory to save model 
parser.add_argument('--resume',     type=bool,default=False,       help='resume training from checkpoint')
parser.add_argument('--start_epoch',type=int, default=0 ,          help='epoch to resume training from')
parser.add_argument('--model_path', type=str, default = '.',       help='the saved model path when you want to resume the training, this should be dir having files of state dict of model and optimizer')
parser.add_argument('--crop_size',  type=int, default=224 ,        help='crop size for image transformation during preprocessing')
parser.add_argument('--lr',         type=float,default=0.00001 ,   help='learning rate for optimizer')
parser.add_argument('--backbone',   type=str,default= 's' ,        help='s:small backbone , b:bigger_backbone')



def create_dataloader(path, batch_size):
  
  data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  image_datasets = {'train': datasets.ImageFolder(os.path.join(path+'/train'),data_transforms['train']),
                   'test': datasets.ImageFolder(os.path.join(path+'/test'),data_transforms['test'])}
  dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size,
                                              shuffle=True, num_workers=4),
              'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size,
                                              shuffle=True, num_workers=4)}

  dataset_sizes = {'train': len(image_datasets['train']), 'test': len(image_datasets['test'])}
  class_names = image_datasets['train'].classes

  return dataloaders, dataset_sizes, class_names


  
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dino model
model_name, out_feature_size, linear_feat_size = ('dinov2_vitb14',int(768),int(512)) if args.backbone=='b' else ('dinov2_vitb14',int(384),int(256)) #feature outputs and inputs of Linear layers are defined her 
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', model_name)  # or dinov2_vits14 is small model 

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, class_names):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out, linear_feat_size),
            nn.BatchNorm1d(linear_feat_size),
            nn.ELU(inplace=True),
            nn.Linear(linear_feat_size, len(class_names)),
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

#load dataloaders
dataloaders, dataset_sizes, class_names = create_dataloader(path = args.data_path, batch_size = args.batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load model and defining loss function and optimizer
model = DinoVisionTransformerClassifier(len(class_names)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.resume:
  model.load_state_dict(torch.load(os.path.join(args.model_path, "model_state_dict.pth")))    # Here you change the file name by model satte dict file name 
  optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer_state_dict.pth"))) # Here you change the file name by optimizer state dict file name

model.to(device)

def validate(model, loss_f, test_loader):
  model.eval()
  correct = 0
  total = 0
  val_loss = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          # calculate outputs by running images through the network
          outputs = model(images.to(device))
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          loss = loss_f(outputs, labels.to(device))
          
          total += labels.size(0)
          correct += (predicted.to("cpu") == labels).sum().item()
          val_loss +=loss.item()

  print(f'Accuracy of the network on the {len(dataloaders["test"])*args.batch_size} test images: {100 * correct // total} %')

  return 100 * (correct // total), val_loss 
  
## Training the model

model.train()
max_val_acc = 0
for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(dataloaders["train"], 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

        if epoch < 5 or epoch > 35:
            val_acc_com, val_loss = validate(model, criterion, dataloaders['test'] )
          
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com 
                model.cpu()
                save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                 }            
                torch.save(save_dict, os.path.join(args.output_path, "vitsb_dino_checkpoint.pth.tar"))
                model.to(device)
              
            with open(args.output_path + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc_com, val_loss))
        else:
            model.cpu()
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                 }            
            torch.save(save_dict, os.path.join(args.output_path, "vitsb_dino_checkpoint.pth.tar"))
            model.to(device)
print('Finished Training')
