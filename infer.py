from models.resnet_3d import get_ResNet,ResNetMLP

import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob 
import os
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index

from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
)
from monai.data import DataLoader, Dataset, CacheDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def predict_func(model, data_path, label_path, device=torch.device("cuda:0")):
    '''inference
    params:
        data_path: 
        label_path: a csv file including survival_status and survival_time
    return:
        c_index, DL risk file
    '''
    clinical = pd.read_csv(label_path)
    path_train_images = sorted(glob(os.path.join(data_path, "*nii.gz")))
    files = [{"image": img} for img in path_train_images]
    df = pd.DataFrame()

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),

            Spacingd(keys=["image"], pixdim=(1,1,1), mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=-125, a_max=225, b_min=0.0, b_max=1.0, clip=True),
            
            Resized(keys=["image"], spatial_size=[112,112,64]),   
            ToTensord(keys=["image"]),  
        ])
    
    model.to(device)

    model.eval()
    for i in range(len(path_train_images)):
        index = test_transforms(files[i])['image_meta_dict']['filename_or_obj'].split("\\")[-1].split(".")[0]
        img = test_transforms(files[i])["image"].unsqueeze(0).to(device)
        DL_feature = model(img).cpu().detach().numpy()

        feature_df=pd.DataFrame(DL_feature,columns=[index],index= ["DL_risk"]).T
        df=df._append(feature_df)
    df.reset_index(drop=False, inplace=True)
    DL_risk_df = pd.concat([clinical[['index','DFS_status','DFS_time']],df.loc[:,["DL_risk"]]], axis=1)

    # calculate c_index
    c_index = concordance_index(DL_risk_df["DFS_time"],DL_risk_df["DL_risk"],DL_risk_df["DFS_status"])
    # print(f'c_index:{c_index}')
    
    return c_index, DL_risk_df

if __name__ == "__main__":
    model = ResNetMLP(model_depth=50, num_classes=1, pretrained=False, freezen_weights=False)
    model_path = r'E:\ccRCC_prognosis\result\finetune_net'
    # model.load_state_dict(torch.load(os.path.join(model_path,"best_metric_train_model.pth")))
    model.load_state_dict(torch.load(os.path.join(model_path,"early_stop_model.pth")))

    print("---------load model completed------------")

    results_dict = dict()
    predict_cohorts = ['train','test','external']
    for i in predict_cohorts:
        clinical_dir = f'E:\\ccRCC_prognosis\\csv\\{i}_clinical.csv'
        data_dir = f'E:\\ccRCC_prognosis\\data\\DL_roi_3d_nii\\{i}'

        c_index, df = predict_func(model=model, data_path=data_dir, label_path=clinical_dir)
        results_dict[i] = c_index

        save_df = os.path.join(model_path, f'{i}_DL.csv')
        df.to_csv(save_df,index=False)
        print(f"-------save completed:{save_df}  ---------")

    results = pd.DataFrame(list(results_dict.items()), columns=['cohort', 'c_index'])
    print(results)