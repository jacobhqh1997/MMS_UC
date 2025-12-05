import random
from tqdm import tqdm
import numpy as np
import torch
import sys
sys.path.append('./')
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    CropForegroundd,
    RandRotated,
    ResizeWithPadOrCropd,
)

import pandas as pd 
import os
from monai.data import CacheDataset
import pickle
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Transform
from lifelines.utils import concordance_index
INFO_PATH = 'path/to/clinical_information/'#clinical info

EPOCH = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cindex_test_max = 0
cindex_binary_max =0
cudnn.deterministic = True
torch.cuda.manual_seed_all(2024)
torch.manual_seed(2024)
random.seed(2024)

from monai.networks.nets import SEResNext50

####################################################
class SEResNext50FeatureExtractor(SEResNext50):
    def __init__(self, layers=(3, 4, 6, 3), groups=32, reduction=16, dropout_prob=None, inplanes=64,
                 downsample_kernel_size=1, input_3x3=False, pretrained=False, progress=True, **kwargs):
        super(SEResNext50FeatureExtractor, self).__init__(
            layers=layers,
            groups=groups,
            reduction=reduction,
            dropout_prob=dropout_prob,
            inplanes=inplanes,
            downsample_kernel_size=downsample_kernel_size,
            input_3x3=input_3x3,
            pretrained=pretrained,
            progress=progress,
            **kwargs
        )
        self.fc = nn.Linear(2048, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
     
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_avg_pool(x)  
        x = torch.flatten(x, 1)
        features = x
        x = self.fc(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear(x)
        return features, x

model = SEResNext50FeatureExtractor(spatial_dims=3, in_channels=1, num_classes=1)
checkpoint = torch.load('path/to/radio_pth.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)

from monai.transforms import Compose
from monai.transforms import Compose, Lambda, MapTransform


val_transforms = Compose(
    [
        LoadImaged(keys=[ "image3"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image3"],
            a_min=0,
            a_max=250,
            b_min=0,######
            b_max=250,
            clip=True,
        ),
        ResizeWithPadOrCropd(
            keys=["image3"],
            spatial_size=(96, 96, 12),
        ),
    ]
)


def prepare_data_list(seg_filepaths, data):
    data_list = []
    for seg_filepath in seg_filepaths:
        
        # print('seg_filepath:',seg_filepath)
        ID = seg_filepath.split('/')[-1][:-13]
        # print('ID:',ID)
        pd_index = data[data['P'].isin([ID])].index.values[0]
        T = data['OS_time'][pd_index]
        O = data['OS_status'][pd_index]
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)
        data_list.append({"image3": seg_filepath, "T": T, "O": O, "seg_filepath":seg_filepath})

    return data_list


def get_Vfiles(path, rule="org.nii.gz"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all



def path_cleaning(macro_path, info_df):
    cleaned_path_V = []
    seg_list_V = get_Vfiles(macro_path)
    # print('seg_list_V:',seg_list_V)
    info_list = list(info_df['P'])
    for i in seg_list_V:
        if os.path.splitext(os.path.basename(i))[0][:-10] in info_list:
            cleaned_path_V.append(i)
    return cleaned_path_V


def filter_values(risk_pred_all, censor_all, survtime_all, file_path_all, wsis_values):
    risk_pred_filtered, censor_filtered, survtime_filtered, file_path_filtered = [], [], [], []
    for risk_pred, censor, survtime, file_path in zip(risk_pred_all, censor_all, survtime_all, file_path_all):
        filename = os.path.splitext(os.path.basename(file_path))[0][:-10]
        # print('filename:',filename)
        if filename in wsis_values:
            # print('filename:',filename)
            risk_pred_filtered.append(risk_pred)
            censor_filtered.append(censor)
            survtime_filtered.append(survtime)
            file_path_filtered.append(filename)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered), file_path_filtered


if __name__ == '__main__':
    macro_test= 'path/to/radio_cropped/'
    info_val =  pd.read_csv('path/to/clinical.csv')
    Val_list = path_cleaning(macro_test, info_val)
    val_data_list = prepare_data_list(Val_list, info_val)
    print('val_data_list:',val_data_list)
    val_dataset = CacheDataset(data=val_data_list, transform=val_transforms, cache_num=10, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    metric_logger = {'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.eval()
        file_path_all = []  
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])  
        for batch_idx, batch_data in enumerate(val_loader):
            image3 = batch_data["image3"].to(device)
    
            survtime = batch_data['T'].to(device)
            censor = batch_data['O'].to(device)
            # weights = batch_data['weight'].type(torch.float).to(device)
            filepath =  batch_data['seg_filepath']
            preds, _ = model(image3)
            filename = os.path.basename(filepath[0]).replace('org.nii.gz', '.pt')
            save_path = os.path.join('path/to/radio/pt/', filename)
            torch.save(preds.cpu(), save_path)
    
