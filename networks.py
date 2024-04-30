import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import BeitForImageClassification
from torchvision import datasets, models, transforms

# EfficientNet-B0 Framework
class EfficientNet(nn.Module):
    def __init__(self,
                num_classes:int = 120,
                model_name:str = ''):
        super().__init__()

        modifying_architecture_mapping = {
            'b0': (1280,32),
            'b1': (1280,32),
            'b2': (1408,32),
            'b3': (1536,40),
            'b4': (1792,48),
            'b5': (2048,48),
            'b6': (2304,56),
            'b7': (2560,64)
        }
        self.in_features = 1280
        self.conv2d_in = 32
        for model_variant, in_features in modifying_architecture_mapping.items():
            if model_variant in model_name:
                self.in_features = in_features[0]
                self.conv2d_in = in_features[1]
        self.model_name = model_name
        self.model = torch.hub.load('pytorch/vision', self.model_name, weights='IMAGENET1K_V1')
        # grayscale img, input channel 1
        self.model.features[0][0] = nn.Conv2d(1, self.conv2d_in, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # output 60 x 2 = 120 for model to predict (x,y) coordinates of the 68 landmarks
        self.model.classifier[1] = nn.Linear(in_features=self.in_features, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x=self.model(x)
        return x
    
# ResnetFramework
class Resnet(nn.Module):
    def __init__(self,
                num_classes:int = 120,
                model_name:str = ''):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision', model_name, weights='IMAGENET1K_V1')
        # grayscale img, input channel 1
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features_mapping = {
            '18':512,
            '34':512,
            '50': 2048,
            '101': 2048,
            '152':2048
        }
        self.in_features = 1280
        for model_variant, in_features in in_features_mapping.items():
            if model_variant in model_name:
                self.in_features = in_features
        # output 60 x 2 = 120 for model to predict (x,y) coordinates of the 68 landmarks
        self.model.fc=nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x


class FusedModule(nn.Module):
    def __init__ (self, in_channels:int = 768, compress_ratio:int = 32):
        super().__init__()
        
        channels = int(in_channels/compress_ratio)
        out_channels = in_channels

        self.lin1 = nn.Linear(2*in_channels, channels)
        self.lin2 = nn.Linear(channels, 2*in_channels)

        self.lin3 = nn.Linear(4*in_channels, out_channels)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(in_channels)
        self.drop = nn.Dropout1d(p=0.4)

    def forward(self, upper_x, lower_x):
        z = torch.cat((upper_x, lower_x), dim=1)
        identity = z
        z1 = self.lin1(z)
        z1 = self.relu(z1)
        z1 = self.drop(z1)

        z1 = self.lin2(z1)
        z1 = self.relu(z1)
        z1 = self.drop(z1)
        return z1

# BEiT's Framework
class ParallelBEiT(nn.Module):
    def __init__(self,
                 num_classes:int = 120,
                 pretrained:str='microsoft/beit-base-patch16-224',
                 variation:str ='plus'):
        super().__init__()
        upper_model = BeitForImageClassification.from_pretrained(pretrained)
        self.variation = variation 
        self.upper_model = torch.nn.Sequential(*(list(upper_model.children())[:-1]))

        lower_model = BeitForImageClassification.from_pretrained(pretrained)
        self.lower_model = torch.nn.Sequential(*(list(lower_model.children())[:-1]))
        if self.variation.__contains__('concat'):
            self.classifier = nn.Linear(in_features=upper_model.classifier.in_features * 2, out_features=120, bias=True)
        elif self.variation.__contains__('module'):
            self.fusedModule = FusedModule()
            self.classifier = nn.Linear(in_features=upper_model.classifier.in_features * 2, out_features=120, bias=True)
        else:
            self.classifier = nn.Linear(in_features=upper_model.classifier.in_features, out_features=120, bias=True)

    def forward(self, original_x, preprocess_x):
        upper_x = self.upper_model(original_x).pooler_output
        lower_x = self.lower_model(preprocess_x).pooler_output
        if self.variation.__contains__('concat'):
            output = self.classifier(torch.cat([upper_x, lower_x], dim=1))
            return output
        elif self.variation.__contains__('module'):
            module_output = self.fusedModule(upper_x, lower_x)
            return self.classifier(module_output) 
        else:
            return self.classifier(upper_x + lower_x)
        
if __name__ == '__main__':
    upper_x = torch.rand([1, 3, 224, 224])
    lower_x = torch.rand([1, 3, 224, 224])

    net = ParallelBEiT()
    # pred = net(upper_x, lower_x)
    print(net)


    

