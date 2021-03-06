import os
from typing import Tuple
import numpy as np
import torch
import cv2
import base64
from model.net import SSNet
from image_helper import image2tensor, mask_to_uint8, normalize_salmap, get_peaksBB, get_candidate_bb, rank_xyxy, project_to_crop_size


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

    seg_uint8 = mask_to_uint8(seg_np)
    sal_uint8 = mask_to_uint8(sal_np)
    
    return seg_uint8, sal_uint8

def model_predict(model: SSNet, img: Mat, device: torch.device) -> Tuple[Mat, Mat]:
    img_tensor = image2tensor(img)
    img_tensor.to(device)
    seg, sal = model.predict(img_tensor)
    seg, sal = postprocess_res2numpy(seg, sal)

    return seg, sal

def crop(sal: Mat, seg: Mat, peaks_bb: np.ndarray, target_size: Size, img_size: Size) -> np.ndarray:
    h_feat, w_feat = seg.shape
    feat_size = (w_feat, h_feat)
    candidate_xyxy = get_candidate_bb(peaks_bb, target_size, img_size, feat_size)
    scores_xyxy = rank_xyxy(candidate_xyxy, sal, seg)
    return scores_xyxy

# def calculate_target_feature_size(img_size: Size, feature_size:Size, crop_size:Size) -> Size:
#     w, h = img_size
#     w_feat, h_feat = feature_size
#     w_ratio = w / w_feat
#     h_ratio = h / h_feat
#     print(w_ratio, h_ratio)
#     crop_w, crop_h = crop_size

#     target_size = (int(crop_w / w_ratio), int(crop_h / h_ratio))
#     return target_size

def calculate_target_feature_size_256(img_size: Size, crop_size:Size) -> Size:
    w, h = img_size
    ratio = max(w,h) / 256
    crop_w, crop_h = crop_size

    target_size = (int(crop_w / ratio), int(crop_h / ratio))
    return target_size

def filter_candidate(scores_xyxy: np.ndarray, alpha: int = 0.1) -> np.ndarray:
    min_score = scores_xyxy[0][0]*alpha
    return [s for s in scores_xyxy if s[0] > min_score]    

def predict_crop(model: SSNet, img: Mat, crop_size: Size, device: torch.device):
    ## model inferencing
    seg, sal = model_predict(model, img, device)

    ## crop preprocessing
    peaks_bb = get_peaksBB(seg, sal)
    # visualize_peak_bb(peaks_bb, seg)

    ## Sizing
    h,w,_ = img.shape
    img_size = (w,h)
    # target_size = calculate_target_feature_size((w,h), (w_feat,h_feat), crop_size)
    target_size = calculate_target_feature_size_256((w,h), crop_size)

    ## Cropping
    scores_xyxy = crop(sal, seg, peaks_bb, target_size, img_size)
    ## filter low score 
    scores_xyxy = filter_candidate(scores_xyxy, 0.1)
    
    ## Cropping postprocessing
    scores_xyxy = project_to_crop_size(scores_xyxy, target_size, crop_size, img_size)

    return scores_xyxy, peaks_bb, seg, sal

def toSalSegBase64(sal: Mat, seg: Mat) -> str:
    """ merge sal seg into 1 3-channel image with R: padding, G: SEG, B: SAL and convert to base64

    Returns:
        str: base64 image string
    """
    sal_norm = mask_to_uint8(normalize_salmap(sal.copy()))
    salseg = np.stack([sal, seg, sal_norm], axis=2)
    retval, buffer_img= cv2.imencode('.png', salseg)
    im_bytes = buffer_img.tobytes()
    b64 = base64.b64encode(im_bytes).decode("utf-8")
    return b64

def decodeSalSegBase64(b64: str) -> Tuple[Mat, Mat]:
    im_bytes = base64.b64decode(b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, -1)
    sal_img = img[:,:,0]
    seg_img = img[:,:,1]
    return  sal_img, seg_img

def crop_b64(b64: str, peaks_bb: np.ndarray, target_size: Size) -> np.ndarray:
    sal, seg = decodeSalSegBase64(b64)
    scores_xyxy = crop(sal, seg, peaks_bb, target_size)
    return scores_xyxy
