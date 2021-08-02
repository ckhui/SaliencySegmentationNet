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

from deploy import loadModel, model_predict, get_peaksBB, crop, toSalSegBase64, decodeSalSegBase64, crop_b64, filter_candidate, project_to_crop_size, calculate_target_feature_size
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

@app.route('/demo', methods=['POST'])
def test():
    if request.method == 'POST':
        print(request.form['crop_x'])
        print(request.form['crop_y'])
        print(request.files['input'])
    return "Hello"

@app.route('/sspredict', methods=['POST'])
def predict():
    try: 
        if request.method == 'POST':
            res_data = {}
            start = time.time()
            if "input" not in request.files:
                res_data['err'] = "Input Error"
                return response_builder(res_data)

            crop_w = request.form.get('crop_w', 0)
            crop_h = request.form.get('crop_h', 0)
            if crop_w == 0 or crop_h == 0:
                res_data['err'] = "Invalid Crop Size, crop size cannot be zero"
                return response_builder(res_data)
            crop_size = (int(crop_w), int(crop_h))


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
            res_data['peaks_bb'] = peak2json(peaks_bb)

            # crop
            h,w,_ = image.shape
            h_feat, w_feat = seg.shape
            target_size = calculate_target_feature_size((w,h), (w_feat,h_feat), crop_size)
            scores_xyxy = crop(sal, seg, peaks_bb, target_size)
            scores_xyxy = filter_candidate(scores_xyxy, 0.1)
            scores_xyxy = project_to_crop_size(scores_xyxy, target_size, crop_size)
            res_data['results'] = score2json(scores_xyxy)
            
            end = time.time()
            res_data['time'] = end-start


            return response_builder(res_data)
    except Exception as err:
        return response_builder({'err': str(err)})

@app.route('/sscrop', methods=['POST'])
def sscrop():
    try:
        if request.method == 'POST':
            res_data = {}
            #  param check 
            start = time.time()
            for k in ['sal_seg', 'peak_bb', 'img_w', 'img_h', 'crop_w', 'crop_h']:
                if k not in request.json:
                    res_data['err'] = f"Missing Param [{k}]"
                    return response_builder(res_data)

            seg_sal_b64 = request.json['sal_seg']
            peaks_json = request.json['peak_bb']
            peaks_bb = read_peaksjson(peaks_json)
            img_w = int(request.json['img_w'])
            img_h = int(request.json['img_h'])
            crop_w = int(request.json['crop_w'])
            crop_h = int(request.json['crop_h'])
            if crop_w == 0 or crop_h == 0:
                res_data['err'] = "Invalid Crop Size, crop size cannot be zero"
                return response_builder(res_data)
            img_size = (img_w, img_h)
            crop_size = (crop_w, crop_h)
            
            seg_sal_b64 = seg_sal_b64.split(',')[-1]
            sal, seg  = decodeSalSegBase64(seg_sal_b64)
            h_feat, w_feat = seg.shape
            target_size = calculate_target_feature_size(img_size, (w_feat,h_feat), crop_size)
            scores_xyxy = crop(sal, seg, peaks_bb, target_size)
            scores_xyxy = filter_candidate(scores_xyxy, 0.1)
            scores_xyxy = project_to_crop_size(scores_xyxy, target_size, crop_size)
            res_data['results'] = score2json(scores_xyxy)

            end = time.time()
            res_data['time'] = end-start
            return response_builder(res_data)
    except Exception as err:
        return response_builder({'err': str(err)})

def read_peaksjson(peaks_json):
    return {tuple(p['peak']) : p['rect'] for p in peaks_json }

def peak2json(peaks_bb):
    peaks_data = [{
        'peak': list(map(int,p)),
        'rect': list(map(int,bb)),
        } for (p,bb) in peaks_bb.items()]
    return peaks_data

def score2json(score_xyxy):
    score_data = [{
        'score': score,
        'rect': list(map(int,xyxy)),
        } for (score,xyxy) in score_xyxy]
    return score_data
        

if __name__ == '__main__':
    print("Starting Server - Loading Model")
    torch.set_flush_denormal(True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # MODEL_WEIGHT = "../../weights/SSNET_best.pth"
    MODEL_WEIGHT = "../../weights/SSNET_ss_0.pth"
    MODEL = loadModel(MODEL_WEIGHT, DEVICE)
    print("Starting Server - Model Loaded")
    
    print("Server Started")
    app.run(debug=True)
    