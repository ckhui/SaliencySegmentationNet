from PIL import Image
from matplotlib.colors import Normalize
from numpy.core.fromnumeric import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class SaliconCoCoDataset(Dataset):
    def __init__(self,dataset_folder, img_name_list, size=256, transform=None):
        self.root = dataset_folder
        self.img_list = img_name_list
        self.transform = transform


        self.map_transfrom= A.Compose([
            A.ToFloat(p=1),
            A.LongestMaxSize(max_size=size, p=1),
            A.PadIfNeeded(
                min_height=size, min_width=size, 
                border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), mask_value=(0), 
                p=1),
            ToTensorV2(p=1),
        ])

        self.raw_transform = A.Compose([
            A.Normalize(
                mean = (0.485, 0.456, 0.406), 
                std = (0.229, 0.224, 0.225),
                p=1),
            A.LongestMaxSize(max_size=size, p=1),
            A.PadIfNeeded(
                min_height=size, min_width=size, 
                border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), mask_value=(0), 
                p=1),
            ToTensorV2(p=1),
        ])    

    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        split = get_split(img_name)
        raw_path = f"{self.root}/images/{split}/{img_name}.jpg" #jpg
        sal_path = f"{self.root}/maps/{split}/{img_name}.png" #png
        obj_path = f"{self.root}/objs/{split}/{img_name}.png" #png

        raw = Image.open(raw_path).convert('RGB') ##[h,w,3]
        sal = Image.open(sal_path).convert('L')   ##[h,w]
        obj = Image.open(obj_path).convert('L')   ##[h,w]

        # TO Numpy
        raw = np.array(raw)
        sal = np.array(sal)
        obj = np.array(obj)

        # print("BEFORE")
        # print(np.max(raw), np.min(raw))
        # print(np.max(sal), np.min(sal))
        # print(np.max(obj), np.min(obj))

        raw = self.raw_transform(image=raw)['image']
        sal = self.map_transfrom(image=sal)['image']
        obj = self.map_transfrom(image=obj)['image']

        # print("AFTER")
        # print(raw.max(), raw.min())
        # print(sal.max(), sal.min())
        # print(obj.max(), obj.min())

        data = {'image':raw, 'saliency':sal, 'mask': obj}
        return data

def get_split(img_name: str) -> str:
    if img_name[5:10] == 'train':
        return 'train'
    else:
        return 'val'

if __name__ == "__main__":
    dataset_folder = "../data/"
    txt_path = "./train_list.txt"
    img_list = np.loadtxt(txt_path, dtype=str)
    dataset = SaliconCoCoDataset(dataset_folder, img_list)

    BATCH_SIZE = 32
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    a = iter(train_loader).next()
    print(a['image'].dtype, a['saliency'].dtype, a['mask'].dtype)
    print(a['image'].shape, a['saliency'].shape, a['mask'].shape)

        