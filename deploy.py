import os
from typing import Tuple
import numpy as np
import torch
from model.net import SSNet
from image_helper import image2tensor, mask_to_uint8, saliency2peaks, get_peaksBB, get_candidate_bb, rank_xyxy, project_to_crop_size

Size = Tuple[int, int]
Mat = np.ndarray

def loadModel(path: str, device: torch.device) -> SSNet: 
    assert os.path.isfile(path), "Invalid Path for Model Weight"

    model = SSNet(3, pretrain_resnet=False)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    return model

def postprocess_res2numpy(seg: torch.Tensor, sal: torch.Tensor) -> Tuple[Mat, Mat]:
    seg_np = seg[0,0].cpu().detach().numpy()
    sal_np = 1 - sal[0,0].cpu().detach().numpy()
    
    return seg_np, sal_np

def model_predict(model: SSNet, img: Mat) -> Tuple[Mat, Mat]:
    img_tensor = image2tensor(img)
    seg, sal = model.predict(img_tensor)
    seg, sal = postprocess_res2numpy(seg, sal)

    return seg, sal

def crop(sal: Mat, seg: Mat, peaks_bb: np.ndarray, target_size: Size) -> np.ndarray:
    h_feat, w_feat = seg.shape
    max_size = (w_feat, h_feat)
    candidate_xyxy = get_candidate_bb(peaks_bb, target_size, max_size)
    scores_xyxy = rank_xyxy(candidate_xyxy, sal, seg)
    return scores_xyxy

def calculate_target_feature_size(img_size: Size, feature_size:Size, crop_size:Size) -> Size:
    w, h = img_size
    w_feat, h_feat = feature_size
    w_ratio = w / w_feat
    h_ratio = h / h_feat
    crop_w, crop_h = crop_size

    target_size = (int(crop_w / w_ratio), int(crop_h / h_ratio))
    return target_size

def filter_candidate(scores_xyxy: np.ndarray, alpha: int = 0.1) -> np.ndarray:
    min_score = scores_xyxy[0][0]*alpha
    return [s for s in scores_xyxy if s[0] > min_score]    

def predict_crop(model: SSNet, img: Mat, crop_size: Size):
    ## model inferencing
    seg, sal = model_predict(model, img)

    ## crop preprocessing
    peaks_bb = get_peaksBB(seg, sal)
    # visualize_peak_bb(peaks_bb, seg)

    ## Sizing
    h,w,_ = img.shape
    h_feat, w_feat = seg.shape
    target_size = calculate_target_feature_size((w,h), (w_feat,h_feat), crop_size)

    ## Cropping
    scores_xyxy = crop(sal, seg, peaks_bb, target_size)
    ## filter low score 
    scores_xyxy = filter_candidate(scores_xyxy, 0.1)
    
    ## Cropping postprocessing
    scores_xyxy = project_to_crop_size(scores_xyxy, target_size, crop_size)

    return scores_xyxy, peaks_bb, seg, sal

