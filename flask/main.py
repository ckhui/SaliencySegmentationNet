from os import EX_OK
import sys
sys.path.append("../")
from PIL import Image
import torch
import time
import numpy as np
import json
import base64
from io import BytesIO

from flask import Flask
from flask import request

from deploy import loadModel, model_predict, get_peaksBB, crop, toSalSegBase64, crop_b64, filter_candidate, project_to_crop_size

app = Flask(__name__)


def read_image_from_bytes(img_bytes):
    try:
        image = Image.open(BytesIO(img_bytes))
        image = np.array(image)
        return image, str(image.shape)
    except:
        return None, "Error Reading Image"

def response_builder(res_data):
    response = app.response_class(mimetype='application/json')
    response.response = json.dumps(res_data)
    response.status = 500 if 'err' in res_data else 200
    return response

@app.route('/')
def hello():
    return f"Model is ready: {MODEL_WEIGHT}"

@app.route('/sspredict', methods=['POST'])
def predict():
    if request.method == 'POST':
        res_data = {}
        start = time.time()
        if "input" not in request.files:
            res_data['err'] = "Input Error"
            return response_builder(res_data)

        img_bytes = request.files["input"].read()
        image, msg = read_image_from_bytes(img_bytes)
        if image is None:
            res_data['err'] = msg
            return response_builder(res_data)
        res_data['img_info'] = msg
        seg, sal = model_predict(MODEL, image)
        salseg = toSalSegBase64(sal,seg)
        res_data['sal_seg'] = salseg
        peaks_bb = get_peaksBB(seg, sal)
        peaks_data = [{
            'peak': list(map(int,p)),
            'rect': list(map(int,bb)),
            } for (p,bb) in peaks_bb.items()]
            
        res_data['peaks_bb'] = peaks_data

        # crop
        # scores_xyxy = filter_candidate(scores_xyxy, 0.1)
        # project_to_crop_size(scores_xyxy, target_size, crop_size)
        
        end = time.time()
        res_data['time'] = end-start


        return response_builder(res_data)
    else:
        return 

@app.route('/sscrop', methods=['POST'])
def sscrop():
    if request.method == 'POST':
        # get seg sal # base64
        # get peaks_bb # array
        # get src_size, crop_size calculate targetsize
        
        # h,w,_ = img.shape
        # h_feat, w_feat = seg.shape
        # target_size = calculate_target_feature_size((w,h), (w_feat,h_feat), crop_size)
        # scores_xyxy = crop(sal, seg, peaks_bb)
        res = {}
        return response_builder(res)


if __name__ == '__main__':
    print("Starting Server - Loading Model")
    torch.set_flush_denormal(True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # MODEL_WEIGHT = "../../weights/SSNET_best.pth"
    MODEL_WEIGHT = "../../weights/SSNET_ss_0.pth"
    MODEL = loadModel(MODEL_WEIGHT, DEVICE)
    print("Starting Server - Model Loaded")
    
    print("Server Started")
    app.run()
    