import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import cv2


from math import *
from networks import Resnet, EfficientNet
from manga_dataset import MangaLandmarksDataset
from transform import Transforms, TransformsNoAugmentation
from utils import *

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    sys.stdout.flush()

# DEFINE DATASET AND DATA LOADER
train_batch_size = 16
val_batch_size = 16
# Change dataset dolder here
dataset_folder = 'dataset'
train_dataset = MangaLandmarksDataset(Transforms(), folder=dataset_folder, split_set='train', isOneCh=True)
valid_dataset = MangaLandmarksDataset(TransformsNoAugmentation(), folder=dataset_folder, split_set='val', isOneCh=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False)

# DEFINE MODEL AND DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_ls = ['resnet18',
              'resnet34',
              'resnet50',
              'resnet101',
              'resnet152',
              'efficientnet_b0',
              'efficientnet_b1',
              'efficientnet_b2',
              'efficientnet_b3',
              'efficientnet_b4',
              'efficientnet_b5',
              'efficientnet_b6',
              'efficientnet_b7',]
selected_weight = 2
save_weight_name = f'{weights_ls[selected_weight]}_'
if save_weight_name.__contains__('resnet'):
    model = Resnet(model_name=weights_ls[selected_weight])
elif save_weight_name.__contains__('efficientnet'):
    model = EfficientNet(model_name=weights_ls[selected_weight])
model.to(device)

# Continue Training Models
# model.load_state_dict(torch.load(f'weights/beit_large_224_sketching_750.pth'))
# print('Weights is loaded from: weights/beit_large_224_sketching_750.pth')

print(f'Using device: {device}')
print(f'Selected dataset: {dataset_folder}')
# print(f'Selected weights: {weights_ls[selected_weight]}')

# DEFINE HYPERPARAMETERS
lr = 0.0001
num_epochs = 1
criterion = nn.MSELoss() # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
      
# DEFINE LOGS'S CONFIGURATION AND ITS FOLDER
attempt = 1
experiment_name = f'{weights_ls[selected_weight]}_{dataset_folder}_{attempt}'
create_directory_if_not_exists(f'exps/{experiment_name}')
log_dir = f'exps/{experiment_name}/logs'
weight_path = f'exps/{experiment_name}/weights'
losses_path = f'exps/{experiment_name}/losses'
figure_path = f'exps/{experiment_name}/figures'
create_directory_if_not_exists(f'{log_dir}')
create_directory_if_not_exists(f'{weight_path}')
create_directory_if_not_exists(f'{losses_path}')
create_directory_if_not_exists(f'{figure_path}')

logging.basicConfig(filename=os.path.join(log_dir, f'{experiment_name}.log'), level=logging.INFO)
logging.info('###########################################')
logging.info(f'Experiment name: {weights_ls[selected_weight]}-{dataset_folder}-{num_epochs}')
logging.info(f'Selected dataset: {dataset_folder}')
logging.info(f'Model: {weights_ls[selected_weight]}')
logging.info(f'Number of epochs: {num_epochs}')
logging.info(f'Learning rate: {lr}')

# TRAINING SESSIONS
train_losses = []
valid_losses = []
start_time = time.time()
for epoch in range(num_epochs):
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    model.train()
    for step, (images, landmarks, filename) in enumerate(train_loader):
        
        images = images.to(device)
        landmarks = landmarks.view(landmarks.size(0),-1).to(device)

        pred = model(images)
        # clear all the gradients before calculating them
        optimizer.zero_grad()

        # find the loss for the current step
        loss_train_step = criterion(pred, landmarks)

        # calculate the gradients
        loss_train_step.backward()

        # update the parameters
        optimizer.step()
        loss_train += loss_train_step.item()
        running_loss = loss_train/(step+1)

        print_overwrite(step+1, len(train_loader), running_loss, 'train')
    
    model.eval()
    with torch.no_grad():
        for step in range(1,len(valid_loader)+1):
            images, landmarks, filename = next(iter(valid_loader))
            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0),-1).to(device)
            predictions = model(images)
            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)
            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step

            print_overwrite(step, len(valid_loader), running_loss, 'valid')
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)

    train_losses.append(loss_train)
    valid_losses.append(loss_valid)

    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch + 1, loss_train, loss_valid))
    print('--------------------------------------------------')

    logging.info('--------------------------------------------------')
    logging.info('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch + 1, loss_train, loss_valid))
    logging.info('--------------------------------------------------')
    # if epoch in interested_epoch:
    if (epoch + 1) % 50 == 0:
        print(f'save weight @{epoch}')
        logging.info(f'save weight @{epoch}')
        torch.save(model.state_dict(), f'{weight_path}/{experiment_name}_{epoch + 1}.pth')


print('Training Complete')
elapsed_time = time.time()-start_time
print(f"Total Elapsed Time : {elapsed_time:.4f} s")

logging.info('Training Complete')
logging.info(f"Total Elapsed Time : {elapsed_time:.4f} s")
logging.info('###########################################')

# STORING TRAIN/VAL LOSSES, VISUALIZATIONS, SEND MESSAGE TO LINE   
save_plot_losses(figure_path, train_losses, valid_losses, experiment_name=experiment_name, stop_epoch=num_epochs)
save_list_to_csv(train_losses, f'{losses_path}/{experiment_name}_train.txt')
save_list_to_csv(valid_losses, f'{losses_path}/{experiment_name}_val.txt')




