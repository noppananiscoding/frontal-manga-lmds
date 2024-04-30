import cv2
import os
import glob
import torch

from math import *
from torch.utils.data import Dataset


class MangaLandmarksDataset(Dataset):

    def __init__(self, 
                 transform=None,
                 folder:str = '',
                 split_set:str = '',
                 isOneCh:bool=False,
                 imgsz:int = None):
        self.root_img_dir = f'data/{folder}'
        self.root_label_dir = 'data/labels'
        self.isOneCh = isOneCh
        self.split_set = split_set
        self.imgsz = imgsz
        # self.annot_filenames = glob.glob(self.root_dir + '/*.json')
        self.image_filenames = []
        self.landmarks = []
        self.transform = transform
        
        for dataset_folder in os.listdir(os.path.join(self.root_img_dir, self.split_set)):
            self.annot_filenames = glob.glob(os.path.join(self.root_label_dir, self.split_set, dataset_folder) + '/*.txt')

            for txt_path in self.annot_filenames:
                file = open(txt_path,'r')
                landmark = file.read()
                landmark = [float(val) for val in landmark.split('\n')]
                landmark = torch.tensor(landmark).view(60, 2)

                self.landmarks.append(landmark)
                self.image_filenames.append(os.path.join(self.root_img_dir, self.split_set, dataset_folder, txt_path.split("/")[-1].split(".")[0]+'.jpg'))

        # self.landmarks = np.array(self.landmarks).astype('float32')
        
        assert len(self.image_filenames) == len(self.landmarks) # check whether number of images and landmarks match

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        if self.isOneCh:
            image = cv2.imread(self.image_filenames[index], 0)
        else:
            image = cv2.imread(self.image_filenames[index])
        if self.imgsz:
            image = cv2.resize(image, (self.imgsz, self.imgsz))
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks)
        
        landmarks = landmarks - 0.5
        return image, landmarks, self.image_filenames[index]

if __name__ == '__main__':
    train_dataset = MangaLandmarksDataset(None, folder='dataset_224', split_set='train')
    test_dataset = MangaLandmarksDataset(None, folder='dataset_224', split_set='test')
    val_dataset = MangaLandmarksDataset(None, folder='dataset_224', split_set='val')

    print(f'Train set length: {len(train_dataset)}')
    assert(len(train_dataset) == len(train_dataset.landmarks))
    print(f'Test set length: {len(test_dataset)}')
    assert(len(test_dataset) == len(test_dataset.landmarks))
    print(f'Val set length: {len(val_dataset)}')
    assert(len(val_dataset) == len(val_dataset.landmarks))

    print(f'Check integity ->')
    print(f'\tNumber of train-images {len(train_dataset.image_filenames)} == Number of train-annotation {len(train_dataset.landmarks)}')
    print(f'\tNumber of test-images {len(test_dataset.image_filenames)} == Number of test-annotation {len(test_dataset.landmarks)}')
    print(f'\tNumber of val-images {len(val_dataset.image_filenames)} == Number of val-annotation {len(val_dataset.landmarks)}')