import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model.net import SSNet
from dataloader import SaliconCoCoDataset
from model.loss import multi_seg_loss_fusion, multi_bce_loss_fusion

## Param
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 3
EVAL_EPOCH = 1
FREEZE_SEG = True
LOG_INTERVAL = 3

LR = 1e-3
WEIGHT_DECAY = 0.01
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.1

## dataset 
dataset_folder = "../data/"
train_txt_path = "./train_list.txt"
train_img_list = np.loadtxt(train_txt_path, dtype=str)[:5]
train_dataset = SaliconCoCoDataset(dataset_folder, train_img_list)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

val_txt_path = "./train_list.txt"
val_img_list = np.loadtxt(val_txt_path, dtype=str)[:3]
val_dataset = SaliconCoCoDataset(dataset_folder, val_img_list)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
print(f"DATASET: {len(train_dataset)} Train {len(val_dataset)} Val")
    
## model
model = SSNet(3)
model_dir = '../weights/SSNET_init.pth'
model.load_state_dict(torch.load(model_dir), strict=False)
if FREEZE_SEG:
    for name, param in model.named_parameters():
        if "_sal" not in name:
            param.requires_grad = False
model.to(DEVICE)

## optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

## train loop
running_loss = 0.0
batch_count = 0
epoch_loss_sal = 0.0
epoch_loss_seg = 0.0
cum_loss_sal = 0.0
cum_loss_seg = 0.0
val_loss_sal = 0.0
val_loss_seg = 0.0

for epoch in range(EPOCH):
    model.train()
    for data in tqdm(train_loader):
        batch_count += 1
        input_img, sal_map, seg_map = data['image'], data['saliency'], data['mask']
        input_img, sal_map, seg_map = input_img.to(DEVICE), sal_map.squeeze(1).to(DEVICE), seg_map.to(DEVICE)

        optimizer.zero_grad()
        seg_out, sal_out = model(input_img, segmentation=not FREEZE_SEG)
        last_sal_loss, sum_sal_loss = multi_seg_loss_fusion(sal_out[0], sal_out[1],sal_out[2],sal_out[3],sal_out[4],sal_out[5],sal_out[6], sal_map)
        loss = sum_sal_loss
        
        epoch_loss_sal += last_sal_loss.item() # Record
        cum_loss_sal += sum_sal_loss.item() # Record
        
        if not FREEZE_SEG:
            last_seg_loss, sum_seg_loss = multi_bce_loss_fusion(seg_out[0], seg_out[1],seg_out[2],seg_out[3],seg_out[4],seg_out[5],seg_out[6], seg_map)
            loss += sum_seg_loss

            epoch_loss_seg += last_seg_loss.item() # Record
            cum_loss_seg += sum_seg_loss.item() # Record

        loss.backward()
        optimizer.step()
        
        if batch_count % LOG_INTERVAL == 1:
            # print BATCHES Progress
            print(f"[BATCH {batch_count:5d}] sal_loss: {cum_loss_sal/LOG_INTERVAL} seg_loss: {cum_loss_seg/LOG_INTERVAL}")
            cum_loss_sal = 0.0
            cum_loss_seg = 0.0
            
    # print EPOCH Progress
    print(f"[EPOCH {epoch:5d}] sal_loss: {epoch_loss_sal/len(train_dataset)} seg_loss: {epoch_loss_seg/len(train_dataset)}")

    if epoch % EVAL_EPOCH == 0:
        model.eval()
        for data in tqdm(val_loader):
            input_img, sal_map, seg_map = data['image'], data['saliency'], data['mask']
            input_img, sal_map, seg_map = input_img.to(DEVICE), sal_map.squeeze(1).to(DEVICE), seg_map.to(DEVICE)

            seg_out, sal_out = model(input_img)
            last_sal_loss, _ = multi_seg_loss_fusion(sal_out[0], sal_out[1],sal_out[2],sal_out[3],sal_out[4],sal_out[5],sal_out[6], sal_map)
            last_seg_loss, _ = multi_bce_loss_fusion(seg_out[0], seg_out[1],seg_out[2],seg_out[3],seg_out[4],seg_out[5],seg_out[6], seg_map)
            val_loss_sal += last_sal_loss.item()
            val_loss_seg += last_seg_loss.item()

        val_loss_sal = val_loss_sal/len(train_dataset)
        val_loss_seg = val_loss_seg/len(train_dataset)
        total_loss = val_loss_sal + val_loss_seg
        print(f"[EVAL {epoch:5d}] sal_loss: {val_loss_sal} seg_loss: {val_loss_seg} total: {total_loss}")
        val_loss_sal = 0
        val_loss_seg = 0
        

