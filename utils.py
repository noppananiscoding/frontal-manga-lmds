import time
import cv2
import os
import random
import numpy as np
import requests
import json
import torch
import csv
import time
import matplotlib.pyplot as plt
from PIL import Image
from imgurpython import ImgurClient
import torchvision.transforms.functional as TF
import matplotlib.image as mpimg
from collections import OrderedDict
from math import *
import datetime


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("ChinContour", (0, 17)),
	("RightEyeBrow", (17, 22)),
	("LeftEyeBrow", (22, 27)),
	("RightEye", (27, 37)),
	("LeftEye", (37, 47)),
	("RightPupil", (47, 48)),
	("LeftPupil", (48, 49)),
    ("Nose", (49, 50)),
	("Mouth", (50, 60)),
])

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
				  (168, 100, 168), (158, 163, 32), (100, 60, 200), 
				  (100, 60, 200), (100, 60, 200), (180, 42, 220)]
		
	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		# check if are supposed to draw the jawline
		if name == "ChinContour":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)
	
	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# return the output image
	return output

def transform_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = Image.fromarray(image)
	image = TF.resize(image, (224, 224))
	image = TF.to_tensor(image)
	image = TF.normalize(image, [0.5], [0.5])
	image = image.unsqueeze(dim=0)
                
def train_report(experiment_name, 
         elapsed_time='None', 
         lr='None', 
         batch_size='None',
         optimizer='Adam', 
         criterion='MSELoss',
         num_epochs='None', 
         graph_path='None'):
    token = 'Jm35XC0m3Da72WFXdu8LC9J4IhH8eE2405WLAOTMIUv'
    payload = {'message' : f"""
    {experiment_name} DONE!
    Total time: {elapsed_time} ms
    Path to graph: {graph_path}
    **Parameters**
    Learning Rate: {lr}
    Batch Size: {batch_size}
    Optimizer: {optimizer}
    Epoch: {num_epochs}
    Loss function: {criterion}
    """}
    r = requests.post('https://notify-api.line.me/api/notify'
                    , headers={'Authorization' : 'Bearer {}'.format(token)}
                    , params = payload
                    , files = {'imageFile': open(graph_path, 'rb')})
    print('Done sending message to line group.!')

def exception_report(event_message):
    token = 'Jm35XC0m3Da72WFXdu8LC9J4IhH8eE2405WLAOTMIUv'
    payload = {'message' :f"An error occurred:, {event_message}"}
    r = requests.post('https://notify-api.line.me/api/notify'
                , headers={'Authorization' : 'Bearer {}'.format(token)}
                , params = payload)
    print('exeption is sent!')
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def save_list_to_csv(data, filename):
    with open(filename, 'w') as file:
        wr = csv.writer(file, delimiter='\n')
        wr.writerow(data)

def save_plot_losses(figure_path, train_losses, valid_losses, experiment_name, stop_epoch):
    fig = plt.figure(figsize=(12, 10))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'{figure_path}/{experiment_name}_{timestamp}.png'
    # Upper-left subplot (ax1)
    plt.title(f"{experiment_name} @{stop_epoch} epoch(es)")
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    fig.tight_layout()
    plt.savefig(output_path)