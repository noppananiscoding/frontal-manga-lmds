import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import torch
import torch.nn as nn
import csv
from utils import *

from manga_dataset import MangaLandmarksDataset
from transform import TransformsNoAugmentation
from collections import OrderedDict
from transformers import BeitForImageClassification
from tqdm import tqdm
from PIL import Image

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


def dataset_prepoessing(landmarks, img_size=224):
    """
    dataset_preprocessing() -> 
        1. +0.5 for all landmarks, due to -0.5 when loading dataset
        2. * imgs to get the exact position of that particular points on the image
    """
    landmarks_noa = (landmarks + 0.5) * img_size
    return landmarks_noa

def preprocessing(img):
    image = Image.fromarray(img)
    # image = TF.resize(image, (224, 224))
    # image = TF.to_tensor(image)
    # image = TF.normalize(image, [0.5], [0.5])
    image = image.unsqueeze(dim=0)
    return image

def postprocessing(pred, imgsz:int=24):
    """
    ret `pred` as the parameters then
    1. +0.5 for all preds-points and * imgsz(224) to get the exact position on the image
    2. .view(-1, 60, 2) to get the x-y coordinates for all landmarks on manga faces
    3. .squeeze() -> reducing array preds dimension
    4. converting to the numpy()
    """
    # print(pred.logits)
    # pred = (pred.cpu() + 0.5) * 224
    pred = (pred + 0.5) * imgsz
    pred = pred.view(-1,60,2)
    pred = pred.squeeze()
    return pred.detach().numpy()

# DEFINE EVALUATION MATRIXS USED FOR EVALUATED THE RESULT FROM MODELS
# EUCLIDEAN DISTANCE
def euclidean_distance(gt, pred):
    return np.sqrt(np.power(pred[0] - gt[0],2) + np.power(pred[1] - gt[1],2))

# MEAN ABSOLUTE ERROR
def MAE(gt, pred, axis=None):
    MAE = []
    for i, (_landmark, _pred) in enumerate(zip(gt, pred)):
        if axis == None:
            MAE.append(abs(euclidean_distance(_landmark.numpy(), _pred)))
        else:
            MAE.append(abs(_landmark[axis] -  _pred[axis]))
    # return round(np.mean(MAE),1)
    return f"{np.mean(MAE):.1f}"

# MEAN SQUARED ERROR
def MSE(gt, pred, axis=None):
    SE = []
    for _landmark, _pred in zip(gt, pred):
        if axis == None:
            SE.append(np.power(euclidean_distance(_landmark.numpy(), _pred),2))
        else:
            SE.append(np.power(_landmark[axis] -  _pred[axis],2))
    # return round(np.mean(SE),1)
    return f"{np.mean(SE):.1f}"

# ROOT MEAN SQUARED ERROR
def RMSE(gt, pred, axis=None):
    SE = []
    for _landmark, _pred in zip(gt, pred):
        if axis == None:
            SE.append(np.power(euclidean_distance(_landmark.numpy(), _pred),2))
        else:
            SE.append(np.power(_landmark[axis] -  _pred[axis],2))
    # return round(np.sqrt(np.mean(SE)),1)
    rmse = np.sqrt(np.mean(SE))
    return f"{rmse:.1f}"

# FAILURE RATE
def failure_rate(gt_ls, pred_ls, chinNormalized_ls, class_length, axis=None):
    _sum = []
    counter = 0
    _fr = 0
    idx = 0 
    for gt, pred in zip(gt_ls, pred_ls):
        if axis == None:
            _sum.append(euclidean_distance(gt.numpy(), pred))
        elif axis != None:
            _sum.append(abs(gt[axis] -  pred[axis]))
        counter += 1

        if counter == class_length:
            chinNormalizedDistance = np.mean(_sum)/chinNormalized_ls[idx]
            if chinNormalizedDistance > 0.0333:
                _fr += 1
            counter = 0
            idx += 1
            _sum = []
    return f"{_fr/(len(gt_ls)/class_length):.3f}"

# variation = 'large-patch16-224-merged-preprocessed-dataset'
def evaluate(evaluation_name, axis=None):
    class_length = dict()
    pred_ls = []
    gt_ls = []
    network.eval()

    # list of groundtruth landmarks coordinations
    landmarks = []
    for image, target, filename in test_dataset:
        landmarks.append(target)

    with torch.no_grad():
        for image, landmark, filename in tqdm(test_dataset):
            image = image.to(device)
            pred = postprocessing(network(image.view(1, 3, imgsz, imgsz)).logits.cpu(), imgsz=imgsz)
            landmark = dataset_prepoessing(landmarks=landmark, img_size=imgsz)
            pred_ls.extend([pred])
            gt_ls.extend([landmark])

    pred_class = dict()
    gt_class = dict()
    class_length = dict()
    gt_class['Overall'] = []
    pred_class['Overall'] = []
    class_length['Overall'] = 60
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        gt_class[name] = []
        pred_class[name] = []

    for gt in gt_ls:
        gt_class['Overall'].extend(gt)
    for pred in pred_ls:
        pred_class['Overall'].extend(pred)

    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        class_length[name] = k-j
        for gt_each in gt_ls:
            gt_class[name].extend(gt_each[j:k])
        for pred_each in pred_ls:
            pred_class[name].extend(pred_each[j:k])

    chin_normalized_ls = []

    for i, coord in enumerate(gt_ls):
        chin_normalized_ls.append(euclidean_distance(coord[:17][0], coord[:17][-1]))

    # class_length = dict()
    data = [
        ["Overall"],
        ["ChinContour"],
        ["RightEyeBrow"],
        ["LeftEyeBrow"],
        ["RightEye"],
        ["LeftEye"],
        ["RightPupil"],
        ["LeftPupil"],
        ["Nose"],
        ["Mouth"],
        ]

    for (i, name) in enumerate(pred_class.keys()):
        data[i].append(MAE(gt_class[name], pred_class[name], axis=axis))
        data[i].append(MSE(gt_class[name], pred_class[name], axis=axis))
        data[i].append(RMSE(gt_class[name], pred_class[name], axis=axis))
        data[i].append(failure_rate(gt_class[name], pred_class[name], chinNormalized_ls=chin_normalized_ls, class_length=class_length[name],  axis=axis))

    head = ["Class", "MAE", "MSE", "RMSE", "Failure Rate"]

    if axis == None:
        csv_filename = os.path.join(evaluation_name, f'{save_model_name}_xy.csv')
    elif axis == 0:
        csv_filename = os.path.join(evaluation_name, f'{save_model_name}_x.csv')
    else:
        csv_filename = os.path.join(evaluation_name, f'{save_model_name}_y.csv')
    data = np.array(data)
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data[:,1:])


# define used variable
dataset_folder = 'dataset'
imgsz = 384
test_dataset = MangaLandmarksDataset(TransformsNoAugmentation(), folder=dataset_folder, split_set='test', imgsz=imgsz)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'beit'
    weights_ls = ['microsoft/beit-base-patch16-224', 
        'microsoft/beit-base-patch16-384',
        'microsoft/beit-large-patch16-224',
        'microsoft/beit-large-patch16-384']
    selected_weight = 1
    variation = weights_ls[selected_weight].split('/')[-1]
    network = BeitForImageClassification.from_pretrained(weights_ls[selected_weight])
    network.classifier = nn.Linear(in_features=network.classifier.in_features, out_features=120, bias=True)
    # LOAD WEIGHTS INTO MODELS
    interested_epoch = 500
    # weights_path = f'weights/{model_name}_224_{interested_epoch}.pth'
    # weights_path = 'weights/beit_large_384_500.pth'
    # network.load_state_dict(torch.load(weights_path))
    network.to(device)
    print(f'Device: {device}')
    # print(f'Load weight@ {weights_path}')

    # CREATE DIRECTORY FOR EVALUATION
    create_directory_if_not_exists('eval')
    save_model_name = weights_ls[selected_weight].split("/")[-1].replace("-", "_")
    evaluation_name = f'{save_model_name}_eval'
    create_directory_if_not_exists(f'eval/{evaluation_name}')

    for i in range(3):
        if i == 0:
            evaluate(f'eval/{evaluation_name}', axis=None)
        elif i == 1:
            evaluate(f'eval/{evaluation_name}', axis=i-1)
        else:
            evaluate(f'eval/{evaluation_name}', axis=i-1)