import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import copy
import pandas as pd 
import os
from monai.data import CacheDataset
from torch.utils.data import Dataset, DataLoader
from Networks.IM_NCTNet import MultiheadContextualGatedAttentionTransformer
from Networks.loss import NegativeLogLikelihoodSurvivalLoss
from sklearn.metrics import roc_curve
from Networks.loss import count_parameters
EPOCH = 100
LR = 5e-6
LAMBDA_COX = 1
LAMBDA_REG = 3e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cindex_test_max = 0
cindex_binary_max =0
cudnn.deterministic = True
torch.cuda.manual_seed_all(2024)
torch.manual_seed(2024)
random.seed(2024)

macro_file_path = 'path/to/Dataset/macro'
radio_file_path = 'path/to/Dataset/radio'
mic_sizes = [2048, 2048]  
model_size = 'medium'
fusion = 'concat'
from lifelines.utils import concordance_index
model = MultiheadContextualGatedAttentionTransformer(mic_sizes=mic_sizes, model_size=model_size,fusion=fusion, device=device)   
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)
# print('model:',model)
optimizer =  torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
print("Number of Trainable Parameters: %d" % count_parameters(model))
from monai.transforms import Compose
from monai.transforms import Compose, Lambda, MapTransform

def regularize_path_weights(model, reg_type=None):
    l1_reg = None
    
    for W in model.module.classifier.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg


nll_loss_fn = NegativeLogLikelihoodSurvivalLoss()


class CustomDataset(Dataset):
    def __init__(self, micro_filepaths, data, macro_file_path, radio_file_path, clinical_data):
        self.micro_filepaths = micro_filepaths
        self.data = data
        self.macro_file_path = macro_file_path
        self.radio_file_path = radio_file_path
        self.clinical_data = clinical_data

    def __len__(self):
        return len(self.micro_filepaths)

    def __getitem__(self, idx):
        micro_filepath = self.micro_filepaths[idx]
        ID = micro_filepath.split('/')[-1][:-12]
        
        pd_index = self.data[self.data['id'].isin([ID])].index.values[0]
        macro_path = os.path.join(self.macro_file_path, str(self.data['id'][pd_index]) + '.pt')
        radio_path = os.path.join(self.radio_file_path, str(self.data['P'][pd_index]) + '.pt')
        macro = torch.load(macro_path, weights_only=True)
        radio = torch.load(radio_path, weights_only=True)
        micro = torch.load(micro_filepath, weights_only=True)
        if isinstance(micro, dict):
            micro = micro['features'] 
        T = self.data['OS_time'][pd_index]
        O = 1 - self.data['OS_status'][pd_index]
        survival_class = self.data['survival_class'][pd_index]

        O = torch.tensor(O).type(torch.FloatTensor).requires_grad_(False)
        T = torch.tensor(T).type(torch.FloatTensor).requires_grad_(False)
        survival_class = torch.tensor(survival_class).type(torch.LongTensor).requires_grad_(False)

        clinical_features = self.clinical_data.iloc[pd_index].values
        clinical_features = torch.tensor(clinical_features).type(torch.FloatTensor).requires_grad_(False)

        return {
            "micro": micro.requires_grad_(False),
            "macro": macro.requires_grad_(False),
            "radio": radio.requires_grad_(False),
            "T": T,
            "O": O,
            "survival_class": survival_class,
            "clinical_features": clinical_features,
            "micro_filepath": micro_filepath
        }

def prepare_data_list(micro_filepaths, data, macro_file_path, radio_file_path, n_classes=4):

    data_list = []
    uncensored_data = data[data['OS_status'] == 1]
    print('uncensored_data:',uncensored_data.shape)
    survival_class, class_intervals = pd.qcut(uncensored_data['OS_time'], q=n_classes, retbins=True, labels=False)
    eps = 1e-7
    class_intervals[-1] = data['OS_time'].max() + eps
    class_intervals[0] = data['OS_time'].min() - eps
    for i in range(n_classes):
        print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
    print(']')
    data['survival_class'], class_intervals = pd.cut(data['OS_time'], bins=class_intervals, retbins=True, labels=False, right=False, include_lowest=True)
    clinical_columns = ['age', 'gender', 'T', 'N', 'M', 'grade', 'size', 'LVI']
    data[clinical_columns] = data[clinical_columns].astype(str)
    clinical_data = pd.get_dummies(data[clinical_columns])
    return CustomDataset(micro_filepaths, data, macro_file_path, radio_file_path, clinical_data)


def get_microfile(path, rule=".pt"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


def path_cleaning(macro_path, info_df):
    cleaned_path_V = []
    seg_list_V =get_microfile(macro_path)
    info_list = list(info_df['id'])
    for i in seg_list_V:
        if os.path.splitext(os.path.basename(i))[0][:-9] in info_list:
            cleaned_path_V.append(i)
    return cleaned_path_V

def filter_values(risk_pred_all, censor_all, survtime_all, file_path_all, wsis_values):
    risk_pred_filtered, censor_filtered, survtime_filtered, file_path_filtered = [], [], [], []
    for risk_pred, censor, survtime, file_path in zip(risk_pred_all, censor_all, survtime_all, file_path_all):
        filename = os.path.splitext(os.path.basename(file_path))[0][:-9]
        # print('filename:',filename)
        if filename in wsis_values:
            # print('filename:',filename)
            risk_pred_filtered.append(risk_pred)
            censor_filtered.append(censor)
            survtime_filtered.append(survtime)
            file_path_filtered.append(filename)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered), file_path_filtered

if __name__ == '__main__':
    micro_train=  'path/to/Dataset/train/micro'
    micro_test= 'path/to/Dataset/valid/micro'
    info_train = pd.read_csv('path/to/clinical_information/train.csv')
    info_val =  pd.read_csv('path/to/clinical_information/valid.csv')

    Train_V = path_cleaning(micro_train,info_train)
    Valid_V = path_cleaning(micro_test,info_val)

    train_dataset = prepare_data_list(Train_V, info_train, macro_file_path, radio_file_path, n_classes=4)
    val_dataset = prepare_data_list(Valid_V, info_val, macro_file_path, radio_file_path, n_classes=4)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=24)

    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        file_path_all = []  
        loss_epoch = 0
        print('train_model_before_weight')
        print(list(model.parameters())[-1])
        for batch_idx, batch_data in enumerate(train_loader):
            micro = batch_data["micro"].to(device)
            macro = batch_data["macro"].to(device)
            radio = batch_data["radio"].to(device)
            clinc = batch_data["clinical_features"].to(device)
            survival_class = batch_data["survival_class"].to(device)
            survtime = batch_data['T'].to(device)
            censor = batch_data['O'].to(device)
            
            filepath =  batch_data['micro_filepath']
            hazards, survs, Y, attention_scores = model(micro, macro, radio, clinc)
  
            loss_cox = nll_loss_fn(hazards, survs, survival_class, censor)
            loss_reg = regularize_path_weights(model=model)
            loss = LAMBDA_COX*loss_cox  + LAMBDA_REG*loss_reg  
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = -torch.sum(survs, dim=1).detach().cpu().numpy()

            risk_pred_all = np.concatenate((risk_pred_all, pred.reshape(-1)))  
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   
            file_path_all += filepath

        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        loss_epoch /= len(train_loader.dataset)
        max_cindex = 0
        best_threshold = 0

        model.eval()
        val_file_path_all = []  
        risk_pred_all, censor_all, survtime_all= np.array([]), np.array([]), np.array([])
        loss_test = 0
        for  batch_idx, val_batch_data in enumerate(val_loader):
            with torch.no_grad():
                val_micro = val_batch_data["micro"].to(device)
                val_macro = val_batch_data["macro"].to(device)
                val_radio = val_batch_data["radio"].to(device)
                val_clinc = val_batch_data["clinical_features"].to(device)
                val_survival_class = val_batch_data["survival_class"].to(device)
                val_survtime = val_batch_data['T'].to(device)
                val_censor = val_batch_data['O'].to(device)
                val_filepath =  val_batch_data['micro_filepath']
                val_hazards, val_survs, val_Y, val_attention_scores = model(val_micro, val_macro, val_radio, val_clinc)

                val_loss_cox = nll_loss_fn(val_hazards, val_survs, val_survival_class, val_censor)  
                val_loss = LAMBDA_COX*val_loss_cox 
                loss_test += val_loss.data.item()   
                val_pred = -torch.sum(val_survs, dim=1).detach().cpu().numpy()
                risk_pred_all = np.concatenate((risk_pred_all, val_pred.reshape(-1)))   
                censor_all = np.concatenate((censor_all, val_censor.detach().cpu().numpy().reshape(-1)))   
                survtime_all = np.concatenate((survtime_all, val_survtime.detach().cpu().numpy().reshape(-1)))   
                val_file_path_all += val_filepath
        
        metrics_train = {
            'Epoch': epoch,
            'Loss': loss_epoch,                   
        }
        metrics_test = {
            'Epoch': epoch,
            'Loss': loss_test,  
        }
        with open('path/to/IM-NCTNet.txt', 'a') as f:
            train_metrics_str = '[Train]\t\t' + ', '.join(['{:s}: {:.4f}'.format(metric, value) for metric, value in metrics_train.items()]) + '\n'
            test_metrics_str = '[Test]\t\t' + ', '.join(['{:s}: {:.4f}'.format(metric, value) for metric, value in metrics_test.items()]) + '\n'
            f.write(train_metrics_str)
            f.write(test_metrics_str)

        save_path = 'path/to/IM-NCTNet'
        if not os.path.exists(save_path): os.makedirs(save_path)
        epoch_idx = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            save_path + '/model_epoch_{}.pkl'.format(epoch))       
        

