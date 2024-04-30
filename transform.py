import random
import numpy as np
import imutils
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image
from math import *

class Transforms():
    def __init__(self,
                 model_name:str = 'None',
                ):
        self.model_name = model_name
        
    # rotation function
    def rotate(self, image, landmarks, angle):
        # randomize transformation angle from given +/- range
        angle = random.uniform(-angle, +angle)

        # create transformation matrix for rotation
        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        # rotate the image
        image = imutils.rotate(np.array(image), angle)

        # also rotate the landmarks by matrix multiplication using rot-matrix
        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5

        return Image.fromarray(image), new_landmarks

    # brightness/saturation re-adjustment
    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def __call__(self, image, landmarks):
        image = Image.fromarray(image)
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=20)
        
        image = TF.to_tensor(image)
        return image, landmarks
    
class TransformsNoAugmentation():
    def __call__(self, image, landmarks):
        image = Image.fromarray(image)

        image = TF.to_tensor(image)
        return image, landmarks