from typing import Dict, Tuple, List, Union
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature.peak import peak_local_max
from matplotlib import pyplot as plt

Mat = np.ndarray
Point = Tuple[int, int]
Rect = Union[np.ndarray, Tuple[int, int, int, int]]
Size = Tuple[int, int]
ScoreRect = List[List[Union[int, Rect]]]

TRANSFORM = A.Compose([
            A.Normalize(
                mean = (0.485, 0.456, 0.406), 
                std = (0.229, 0.224, 0.225),
                p=1),
            A.LongestMaxSize(max_size=256, p=1),
            ToTensorV2(p=1),
        ])    

def image2tensor(img: Mat) -> torch.Tensor:
    """ Convert RGB image to model input format with preprocessing transformation
    Args:
        img (Mat): RGB image

    Returns:
        torch.Tensor: model input
    """
    img = TRANSFORM(image=img)['image']
    return img.unsqueeze(0)

def thresh(img: Mat) -> Mat:
    """ image to binary mask with OTSU thresholding and erode+dilate

    Args:
        img (Mat): uint8 image

    Returns:
        Mat: Binary Mask
    """
    th, thresh_binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)
    thresh_mask = cv2.erode(thresh_binary,kernel,iterations = 2)
    thresh_mask = cv2.dilate(thresh_mask,kernel,iterations = 2)
    return thresh_mask

def mask_to_uint8(img: Mat) -> Mat:
    """ convert normalized image [0,1] to uint8 [0,255]

    Args:
        img (Mat): image in range [0,1]

    Returns:
        Mat: image in range [0,255]
    """
    if img.max() <= 1:
        img = img*255
    return img.astype(np.uint8)

def saliency2peaks(sal_input: Mat) -> np.ndarray:
    """ get peak from saliency blob
    Saliency -> thresh -> binary
    binary -> find Countour
    Each Countour -> Countour Mask -> Find Peak

    Args:
        sal_input (Mat): uint8 image

    Returns:
        np.ndarray: Peak Points
    """
    peaks = []
    sal_binary = thresh(sal_input)
    contours, hierarchy = cv2.findContours(sal_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        mask = np.zeros_like(sal_binary)
        contour_mask = cv2.drawContours(mask, contours, c, 255, -1)
        fixation = cv2.bitwise_and(sal_input, sal_input, mask=contour_mask)
        peak = peak_local_max(fixation, num_peaks=1)
        if len(peak) > 0:
            peaks.append(peak[0])

    return np.array(peaks, dtype=int)

def peaks2rect(peaks: np.ndarray, seg: Mat) -> Dict[Point, Rect]:
    """ match each peaks to a object blob
    Segmentation -> thresh -> binary
    binary -> find Countour
    Each Peak -> Gather all nearby countours -> rect

    Args:
        peaks (np.ndarray): peak points
        seg (Mat): uint8 segmentation image

    Returns:
        dict[Point, Rect]: mapping for peak point to corresponding rect
    """
    seg_contours, _ = cv2.findContours(thresh(seg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    peaks_map = {}
    for p in peaks:
        points = np.array([[ [ p[1], p[0]] ]])
        for c in seg_contours:
            dist = cv2.pointPolygonTest(c, (int(p[1]), int(p[0])), measureDist=True)
            # store all point with dist > -10, include the peak -> get the bounding box
            if dist > -10:
                points = np.vstack([points,c])
        if len(points) == 1: ## single point
            rect = (p[1], p[0], 1, 1)
        else:
            rect = cv2.boundingRect(points)
        peaks_map[(p[1],p[0])] = rect
    return peaks_map

def bb2xyxy(bb: Rect) -> Rect:
    """ (x,y,w,h) to (x,y,x,y)

    Args:
        bb (Rect): (x,y,w,h) rect

    Returns:
        Rect: (x,y,x,y) rect
    """
    x,y,w,h = bb
    return (x, y, x+w, y+h)

def get_peaksBB(seg: Mat, sal: Mat) -> Dict[Point, Rect]:
    """ croping preporcessing, get peak,rect mapping from saliency and segmentation map

    Args:
        seg (Mat): segmentation map (model output)
        sal (Mat): saliency map (model output)

    Returns:
        Dict[Point, Rect]: mapping for peak point to corresponding rect
    """
    peaks = saliency2peaks(sal)
    peaks_bb = peaks2rect(peaks, seg)

    return peaks_bb

def refine_saliency_aware_xyxy(center: Point, bb: Rect, target_size: Size) -> Rect:
    """ refine saliency aware bounding box

    Args:
        center (Point): peak point
        bb (Rect): peak bounding box in [x,y,w,h]
        target_size (Size): target size

    Returns:
        Rect: output rect in target_size
    """
    cen_x, cen_y = center
    x1,y1,x2,y2 = bb2xyxy(bb)
    tar_x, tar_y = target_size

    fill_x1 = min(cen_x - x1, tar_x//2)
    remainder_x = tar_x - fill_x1
    fill_x2 = min(x2 - cen_x, remainder_x)
    remainder_x -= fill_x2
    if remainder_x > 0:
        fill_x1 += remainder_x//2
        fill_x2 += (remainder_x+1)//2

    fill_y1 = min(cen_y - y1, tar_y//2)
    remainder_y = tar_y - fill_y1
    fill_y2 = min(y2 - cen_y, remainder_y)
    remainder_y -= fill_y2
    if remainder_y > 0:
        fill_y1 += remainder_y//2
        fill_y2 += (remainder_y+1)//2

    return cen_x-fill_x1, cen_y-fill_y1, cen_x+fill_x2, cen_y+fill_y2

def refine_boundary_aware_xyxy(xyxy: Rect, max_size: Size) -> Rect:
    """ refine rect within image boundary

    Args:
        xyxy (Rect): rect in [xyxy]
        max_size (Size): image boundary

    Returns:
        Rect: refined rect
    """
    x1,y1,x2,y2 = xyxy
    w, h = max_size
    diff_x = x2-w
    if diff_x > 0:
        x2 -= diff_x
        x1 -= diff_x
        x1 = max(0, x1)
    elif x1 < 0:
        x2 = min(w, x2-x1)
        x1 = 0

    diff_y = y2-h
    if diff_y > 0:
        y2 -= diff_y
        y1 -= diff_y
        y1 = max(0, y1)
    elif y1 < 0:
        y2 = min(h, y2-y1)
        y1 = 0

    return x1,y1,x2,y2

def get_candidate_bb(peaks_bb: Dict[Point, Rect], target_size: Size, max_size: Size = (256,256)) -> List[Rect]:
    """[summary]

    Args:
        peaks_bb (Dict[Point, Rect]): peak-bb dict
        target_size (Size): target feature size, projected crop size 
        max_size (Size, optional): Max feature size, size of model output. Defaults to (256,256).

    Returns:
        List[Rect]: candidate crop in feature dimension
    """

    rects = []
    for (peak, bb) in peaks_bb.items():
        xyxy = refine_saliency_aware_xyxy(peak,bb,target_size)
        xyxy = refine_boundary_aware_xyxy(xyxy, max_size)
        rects.append(xyxy)
    return list(set(rects))

def normalize_salmap(mat: Mat) -> Mat:
    """ Nomalized Saliency for score calculation

    Args:
        mat (Mat): saliency (model output)

    Returns:
        Mat: normalized saliency
    """
    mat = mat - mat.min()
    mat = mat / mat.max()
    return mat 

def rank_xyxy(candidate_xyxy: List[Rect], sal: Mat, seg: Mat) -> ScoreRect:
    """ rank candidate crop based on sal_score and seg_score
    sal_score = sum of pixel's square
    seg_score = sum of pixel
    total_score = sal_score * seg_score

    Args:
        candidate_xyxy (List[Rect]): Candidate Rect
        sal (Mat): Saliency Map (model output)
        seg (Mat): Segmentation Map (model output)

    Returns:
        ScoreRect: Ranked Score_Rect
    """
    sal_norm = normalize_salmap(sal)
    seg_norm = normalize_salmap(seg)
    scores_xyxy = []
    for xyxy in candidate_xyxy:
        x1,y1,x2,y2 = xyxy
        area = (y2-y1)*(x2-x1)

        sal_partial = sal_norm[y1:y2, x1:x2]
        score_sal = np.power(sal_partial, 2).sum() / area

        seg_partial = seg_norm[y1:y2, x1:x2]
        score_seg = seg_partial.sum() / area
        total = score_sal * score_seg
        # print("sal",score_sal, "seg", score_seg, total)
        scores_xyxy.append([total, xyxy])
    return sorted(scores_xyxy, reverse=True)

def project_to_crop_size(scores_xyxy: ScoreRect, target_size: Size, crop_size: Size) -> ScoreRect:
    """ project the croping result in feature dimension to original image dimension

    Args:
        scores_xyxy (ScoreRect): results with box in feature dimension
        target_size (Size): target size in feature dimension
        crop_size (Size): original target croping size

    Returns:
        ScoreRect: results with box in original dimension
    """
    w_crop, h_crop = crop_size
    w, h = target_size
    w_ratio = w_crop / w
    h_ratio = h_crop / h

    def project_to_crop(xyxy):
        x1,y1,x2,y2 = xyxy
        # print("Final Size [ORI]:", x2-x1, y2-y1)
        x1 = int(x1 * w_ratio)
        y1 = int(y1 * h_ratio)
        x2 = int(x2 * w_ratio)
        y2 = int(y2 * h_ratio)
        # print("Final Size [Before]:", x2-x1, y2-y1)
        if x2-x1 != w_crop:
            adj_x = x2-x1-w_crop
            x1 += adj_x//2
            x2 -= (adj_x+1)//2

        if y2-y1 != h_crop:
            adj_y = y2-y1-h_crop
            y1 += adj_y//2
            y2 -= (adj_y+1)//2

        # print("Final Size [After]:", x2-x1, y2-y1)
        return (x1,y1,x2,y2)

    scores_xyxy_raw = [(s, project_to_crop(rect)) for (s,rect) in scores_xyxy]
    return scores_xyxy_raw

# Visualize
def visualize_peak_bb(peak_map: Dict[Point, Rect], seg: Mat) -> None:
    cm = plt.get_cmap()
    seg_img = (cm(seg)[:,:,:3]* 255).astype(np.uint8)

    for peak, rect in peak_map.items():
        py, px = peak[0], peak[1]
        seg_img[px-2:px+3, py-2:py+3, 0] = 255
        seg_img[px-2:px+3, py-2:py+3, 1] = 0
        seg_img[px-2:px+3, py-2:py+3, 2] = 0
        x,y,w,h = rect
        seg_img = cv2.rectangle(seg_img,(x,y),(x+w,y+h),(255,0,0),2)
    plt.imshow(seg_img)

def visualize_prediction(img: Mat, sal: Mat, seg: Mat, peaks_bb: Dict[Point, Rect]) -> None:
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.subplot(1,4,2)
    plt.imshow(sal)
    plt.subplot(1,4,3)
    plt.imshow(seg)
    plt.subplot(1,4,4)
    visualize_peak_bb(peaks_bb, seg)

def visualize_crop_result(scores_xyxy: ScoreRect, img: Mat) -> None:
    plt.subplot(2,4,1)
    plt.imshow(img)

    for i, (score,xyxy) in enumerate(scores_xyxy):
        if i == 3:
            print(len(scores_xyxy) - 2, "More ...")
            return 
        plt.subplot(2,4,i+2)
        x1,y1,x2,y2 = xyxy
        plt.imshow(img[y1:y2, x1:x2])
        plt.title(score)  
