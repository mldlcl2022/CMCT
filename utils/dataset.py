from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision import transforms
import torch
from PIL import Image

class CMCTDataset(Dataset):
    def __init__(self, data_name, split, csv_dir, root_dir):
        # load csv file
        self.csv_file = pd.read_csv(
            os.path.join(
                csv_dir, f'{split}.csv'
            )
        )
        
        # image file path list
        self.img_list = [
            os.path.join(
                root_dir, data_name, filepath.replace('records500/','')+'-0.png'
            ) for filepath in self.csv_file.filepath
        ]
        
        # task ground truth
        if 'cpsc' in data_name:
            label_start_col = 7
            label_end_col = concept_start_col = label_start_col + 9
        elif 'ptbxl' in data_name:
            label_start_col = 5
            label_end_col = concept_start_col = label_start_col + 5
        self.task_gts = self.csv_file.iloc[:, label_start_col:label_end_col].to_numpy(dtype= 'float32')
        
        # concept ground truth
        self.concept_gts = self.csv_file.iloc[:, concept_start_col:].to_numpy(dtype= 'float32')
        
        # image transform
        self.crop_area = (0, 530, 2200, 1700-70)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        # task label
        task_gt = torch.from_numpy(self.task_gts[idx])
        
        # concept label
        concept_gt = torch.from_numpy(self.concept_gts[idx])
        
        # input image data
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = img.crop(self.crop_area)
        img = self.transforms(img)
        
        return img, concept_gt, task_gt

class ImageDataset(Dataset):
    def __init__(self, data_name, split, csv_dir, root_dir):
        # load csv file
        self.csv_file = pd.read_csv(
            os.path.join(
                csv_dir, f'{split}.csv'
            )
        )
        
        # image file path list
        self.img_list = [
            os.path.join(
                root_dir, data_name, filepath.replace('records500/','')+'-0.png'
            ) for filepath in self.csv_file.filepath
        ]
        
        # task ground truth
        if 'cpsc' in data_name:
            label_start_col = 7
            label_end_col = concept_start_col = label_start_col + 9
        elif 'ptbxl' in data_name:
            label_start_col = 5
            label_end_col = concept_start_col = label_start_col + 5
        self.task_gts = self.csv_file.iloc[:, label_start_col:label_end_col].to_numpy(dtype= 'float32')
        
        # image transform
        self.crop_area = (0, 530, 2200, 1700-70)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        # task label
        task_gt = torch.from_numpy(self.task_gts[idx])
        
        # input image data
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = img.crop(self.crop_area)
        img = self.transforms(img)
        
        return img, task_gt