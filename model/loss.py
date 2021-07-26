'''
KLD and CC adapt from: https://github.com/samyak0210/saliency/blob/master/SimpleNet/loss.py
bce_ssim_loss adapt from: 
'''
# Saliency = KLD + CC
# Segmentation = BCE + SSIM + IOU

import torch
import torch.nn as nn
from . import pytorch_ssim
from . import pytorch_iou


def kldiv(s_map, gt):
    if len(s_map.size()) == 4:
        s_map = s_map.squeeze(1)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    assert expand_gt.size() == gt.size()

    s_map = s_map/(expand_s_map*1.0)
    gt = gt / (expand_gt*1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt/(s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))

def cc(s_map, gt):
    if len(s_map.size()) == 4:
        s_map = s_map.squeeze(1)

    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa*bb)))

def loss_sal(pred_map, gt, kldiv_coeff=1, cc_coeff=1):
    loss = kldiv_coeff * kldiv(pred_map, gt) + cc_coeff * cc(pred_map, gt)
    return loss

def multi_seg_loss_fusion(d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss1 = loss_sal(d1,labels_v)
	loss2 = loss_sal(d2,labels_v)
	loss3 = loss_sal(d3,labels_v)
	loss4 = loss_sal(d4,labels_v)
	loss5 = loss_sal(d5,labels_v)
	loss6 = loss_sal(d6,labels_v)
	loss7 = loss_sal(d7,labels_v)

	loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

	return loss1, loss

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):
	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out
	return loss

def multi_bce_loss_fusion(d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss1, loss

def bce_ssim_iou_individual_loss(pred,target):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)
    return (bce_out, ssim_out, iou_out)