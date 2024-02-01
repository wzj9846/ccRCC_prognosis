import os
from glob import glob
import shutil
# import dicom2nifti
import numpy as np
import pandas as pd
import nibabel as nib
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CenterSpatialCropd,
    RandRotated,
    RandZoomd,
    RandFlipd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
    Rand2DElasticd
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism



def prepare(in_dir, pixdim=(1, 1, 1), a_min=-125, a_max=225, spatial_size=[224,224,112], batch_size=1, cache=False):

    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you 
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    # set_determinism(seed=0)

    path_train_images = sorted(glob(os.path.join(in_dir, 'train', "*image.nii.gz")))
    path_train_label = glob(os.path.join(in_dir, 'train', "*.csv"))[0]
    label_train = pd.read_csv(path_train_label)
    e_train = np.array(label_train['DFS_status'],dtype=np.int64)
    y_train = np.array(label_train['DFS_time'],dtype=np.float32)

    path_test_images = sorted(glob(os.path.join(in_dir, 'test', "*image.nii.gz")))
    path_test_label = glob(os.path.join(in_dir, 'test', "*.csv"))[0]
    label_test = pd.read_csv(path_test_label)
    e_test = np.array(label_test['DFS_status'],dtype=np.int64)
    y_test = np.array(label_test['DFS_time'],dtype=np.float32)

    train_files = [{"image": img, "event": e, "time": y} for img, e, y in zip(path_train_images, e_train,y_train)]
    test_files = [{"image": img, "event": e, "time": y} for img, e, y in zip(path_test_images, e_test,y_test)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 

            RandRotated(keys=['image'], prob=0.5, range_z=[-15,15], keep_size=True,mode=("bilinear"),padding_mode='reflection'),
            RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9,max_zoom=1.1, mode=("area")),
            # RandFlipd(keys=["image"], prob=0.5),
            # RandAffined(keys=["image"], prob=0.6, shear_range=(0.2,0.2), mode=["bilinear"]),
            # Rand2DElasticd(keys=["image"],prob=0.5,spacing=(20,20),magnitude_range=(1,2)),
            # RandGaussianNoised(keys='image', prob=0.1,mean=0.0, std=0.1),

            Resized(keys=["image"], spatial_size=spatial_size),   
            ToTensord(keys=["image"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
            
            Resized(keys=["image"], spatial_size=spatial_size),   
            ToTensord(keys=["image"]),  
        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1)
        train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1)
        test_loader = DataLoader(test_ds, batch_size=batch_size,shuffle=False)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size,shuffle=False)

        return train_loader, test_loader
    
if __name__=="__main__":
    indir = r'E:\TASK_MIBC_ct_prognosis_AI\data\max_axial_ct_v_2d'
    dl_train, dl_test = prepare(indir,batch_size=16)
    for test_data in dl_test:  
        x = test_data['image']
        e = test_data['event']
        y = test_data['time']
        print(x.shape,e.shape,y.shape)
        # print(f"status:{e}, duration:{y}")
