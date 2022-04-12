# . flask/bin/activate
# export FLASK_APP="/Users/xdd44/Dropbox/Final/Architectural_style_transfer/index.py"
# export PYTHONPATH="/Users/xdd44/Dropbox/Final/Architectural_style_transfer/DiverseDepth"

from flask import Flask, request
from flask import render_template
from werkzeug.utils import secure_filename

from styleNames import *

from fastai.vision import *
import torch
from torchvision.transforms import transforms

learn = load_learner("Models")

sys.path.append("/Users/xdd44/Dropbox/Final/Architectural_style_transfer/DiverseDepth")
# export PYTHONPATH="/home/xdd44/Dropbox/Final/Architectural_style_transfer/DiverseDepth"
from Minist_Test.lib.diverse_depth_model import RelDepthModel
from Minist_Test.lib.net_tools import strip_prefix_if_present
from styleNames import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import torch
import dill

depth_model = RelDepthModel()
depth_model.eval()
checkpoint = torch.load("/Users/xdd44/Dropbox/Final/Architectural_style_transfer/Models/diverseDepth.pth", map_location=lambda storage, loc: storage, pickle_module=dill)
model_state_dict_keys = depth_model.state_dict().keys()
checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

if all(key.startswith('module.') for key in model_state_dict_keys):
	depth_model.module.load_state_dict(checkpoint_state_dict_noprefix)
else:
	depth_model.load_state_dict(checkpoint_state_dict_noprefix)
del checkpoint

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("demo.html")

@app.route('/transfer', methods = ['GET'])
def transfer():
	styleIndex = int(request.args.get('style'))
	print(styleNames[styleIndex])
	return styleNames[styleIndex]

@app.route('/predict', methods = ['GET'])
def predict():
	fileName = request.args.get('fileName')[12:]
	dir = "/Users/xdd44/Documents/CS4514/Demo/"
	image = open_image(dir + fileName)
	pred_class, pred_idx, outputs = learn.predict(image)
	print(pred_class)
	generateDepth(dir, fileName)
	return str(pred_class)

def generateDepth(dir, fileName):
	rgb = cv2.imread(dir + fileName)
	rgb_c = rgb[:, :, ::-1].copy()
	A_resize = cv2.resize(rgb_c, (385, 385))
	img_torch = scale_torch(A_resize)[None, :, :, :]
	pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
	pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
	plt.imsave(os.path.join(dir + "Depth/depth.jpg"), pred_depth_ori, cmap='rainbow')

def scale_torch(img):
	if len(img.shape) == 2:
		img = img[np.newaxis, :, :]
	if img.shape[2] == 3:
		transform = transforms.Compose([transforms.ToTensor(),
										transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
		img = transform(img)
	else:
		img = img.astype(np.float32)
		img = torch.from_numpy(img)
	return img

if __name__ == "__main__":
	app.run(debug=True)