from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
import numpy as np
import re
import random

import warnings
warnings.filterwarnings("ignore")

class MyDataset(Dataset):
    def __init__(self, path_dir, transform=None, train=True, test=True, val=True) -> None:
        self.path_dir = path_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(self.path_dir, "photos"))
        self.train = train
        self.test = test
        self.val = val

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> any:
        category = None
        tiller_num = None
        img = self.images[index]

        csv_path = os.path.join(self.path_dir, "test_2.csv")
        df = pd.read_csv(csv_path)
        column_name = df.applymap(lambda x: x == img).any().idxmax()   #from elemant content search its column name



        if hasattr(re.search(r'_(.*?)_', column_name), 'group'):  
            # 如果属性存在，则安全地访问它
            print(f"{img} OKOKOKOKOKOKOK")
            day_number = re.search(r'_(.*?)_', column_name).group(1) 
            # 接下来使用value进行其他操作...  
        else:  
            # 如果属性不存在，则处理异常或记录错误  
            print(f"对象 {img}  ERROR!")  
            # 可以选择抛出异常、记录日志或进行其他错误处理
        # day_number = re.search(r'_(.*?)_', column_name).group(1) # from column name get its day number(to find if useful)



        row_indice = np.where(df[column_name] == img)[0][0] # from column name get its row indice

        # category = df.species[row_indice]
        tiller_column_name = "day_" + day_number
        tiller_num = df[tiller_column_name][row_indice]
        
        image = Image.open(os.path.join(self.path_dir, "photos", img))
        image_tensor = self.transform(image)

        return image_tensor, tiller_num
    


if __name__ == '__main__':
    mytransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = MyDataset(path_dir="./datasets", transform=mytransform)
    image, tiller_num = dataset[0]
    print(image.shape)

    # (img, category) -> tiller_num
    # (img, category, (day0, day4 ...)) -> future tiller_num
    #task: not CLASSIFICATION, but REGRESSION