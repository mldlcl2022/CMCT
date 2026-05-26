from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision import transforms
import torch
from PIL import Image

class CMCTDataset(Dataset):
    def __init__(self, data_name, split, csv_dir, root_dir):
        # load csv file #
        self.csv_file = pd.read_csv(
            os.path.join(
                csv_dir, data_name, f'{split}.csv'
            )
        )
        
        # image filepath list #
        self.img_list = [
            os.path.join(
                root_dir, data_name.split('_')[0], filepath.replace('records500/','')+'-0.png'
            ) for filepath in self.csv_file.filepath
        ]
        
        # task ground truth #
        if 'ptbxl' in data_name:
            label_start_col = 5
            if 'super' in data_name:
                label_end_col = label_start_col + 5
            elif 'sub' in data_name:
                label_end_col = label_start_col + 23
            elif 'rhythm' in data_name:
                label_end_col = label_start_col + 12
            elif 'form' in data_name:
                label_end_col = label_start_col + 19
        elif data_name == 'cpsc':
            label_start_col = 7
            label_end_col   = label_start_col + 9
        self.task_gts = self.csv_file.iloc[:,label_start_col:label_end_col].to_numpy(dtype= 'float32')
        
        # concept ground truth #
        self.concept_gts = self.csv_file.iloc[:,label_end_col:].to_numpy(dtype= 'float32')
        
        # image transform #
        self.crop_area = (0, 530, 2200, 1700-70)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        # task label #
        task_gt = torch.from_numpy(self.task_gts[idx])
        
        # concept label #
        concept_gt = torch.from_numpy(self.concept_gts[idx])
        
        # image data #
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = img.crop(self.crop_area)
        img = self.transform(img)
        
        return img, concept_gt, task_gt

# class ImageDataset(Dataset):