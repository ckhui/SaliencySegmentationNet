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

from deploy import loadModel, model_predict, get_peaksBB, predict_crop, toSalSegBase64

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
        sal, seg = model_predict(MODEL, image)
        salseg = toSalSegBase64(sal,seg)
        res_data['sal_seg'] = salseg
        peaks_bb = get_peaksBB(seg, sal)
        peaks_data = [{
            'peak': list(map(int,p)),
            'rect': list(map(int,bb)),
            } for (p,bb) in peaks_bb.items()]
            
        res_data['peaks_bb'] = peaks_data
        end = time.time()
        res_data['time'] = end-start

        return response_builder(res_data)
    else:
        return 

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
    