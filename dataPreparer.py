import sys
sys.path.append('/home/xdd44/Dropbox/Final/Architectural_style_transfer/DiverseDepth')
# export PYTHONPATH="/home/xdd44/Dropbox/Final/Architectural_style_transfer/DiverseDepth"
from Minist_Test.lib.diverse_depth_model import RelDepthModel
from Minist_Test.lib.net_tools import strip_prefix_if_present
from styleNames import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch
import dill

def scale_torch(img):
	"""
	Scale the image and output it in torch.tensor.
	:param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
	:param scale: the scale factor. float
	:return: img. [C, H, W]
	"""
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


if __name__ == '__main__':

	updateMirror = False
	updateDepth = False

	# create depth model
	depth_model = RelDepthModel()
	depth_model.eval()

	# load checkpoint
	checkpoint = torch.load("DiverseDepth/model.pth", map_location=lambda storage, loc: storage, pickle_module=dill)
	model_state_dict_keys = depth_model.state_dict().keys()
	checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

	if all(key.startswith('module.') for key in model_state_dict_keys):
		depth_model.module.load_state_dict(checkpoint_state_dict_noprefix)
	else:
		depth_model.load_state_dict(checkpoint_state_dict_noprefix)
	del checkpoint
	torch.cuda.empty_cache()
	depth_model.cuda()

	if (updateMirror):
		count = 0
		originDir = "../Dataset/Photos/Origin"
		mirrorDir = "../Dataset/Photos/Mirror"
		for styleName in os.listdir(originDir):
			styleDir = os.path.join(originDir, styleName)
			if (os.path.isdir(styleDir)):
				for arcName in os.listdir(styleDir):
					arcDir = os.path.join(styleDir, arcName)
					if (os.path.isdir(arcDir)):
						for photoName in os.listdir(arcDir):
							photoDir = os.path.join(arcDir, photoName)
							if (os.path.isfile(photoDir)):
								count += 1
								if (count % 100 == 0):
									print("processing (%d)-th image..." % (count))
								try:
									origin = cv2.imread(photoDir)
									mirror = cv2.flip(origin, 1)
									os.makedirs(os.path.join(mirrorDir, styleName, arcName), exist_ok=True)
									cv2.imwrite(os.path.join(mirrorDir, styleName, arcName, photoName), mirror)
								except:
									print("Error in %s %s %s" % (styleName, arcName, photoName))
	if (updateDepth):
		count = 0
		photosDir = "../Dataset/Photos"
		depthsDir = "../Dataset/Depths"
		for fileName in os.listdir(photosDir):
			print(fileName)
			if (fileName == "Origin"):
				continue
			fileDir = os.path.join(photosDir, fileName)
			if (os.path.isdir(fileDir)):
				for styleName in os.listdir(fileDir):
					styleDir = os.path.join(fileDir, styleName)
					if (os.path.isdir(styleDir)):
						for arcName in os.listdir(styleDir):
							arcDir = os.path.join(styleDir, arcName)
							if (os.path.isdir(arcDir)):
								for photoName in os.listdir(arcDir):
									photoDir = os.path.join(arcDir, photoName)
									if (os.path.isfile(photoDir)):
										count += 1
										if (count % 100 == 0):
											print("processing (%d)-th image..." % (count))
										try:
											rgb = cv2.imread(photoDir)
											rgb_c = rgb[:, :, ::-1].copy()
											A_resize = cv2.resize(rgb_c, (385, 385))
											img_torch = scale_torch(A_resize)[None, :, :, :]
											pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
											pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
											os.makedirs(os.path.join(depthsDir, fileName, styleName, arcName), exist_ok=True)
											plt.imsave(os.path.join(depthsDir, fileName, styleName, arcName, photoName), pred_depth_ori, cmap='rainbow')
										except:
											print("Error in %s %s %s" % (styleName, arcName, photoName))
	
	count = 0
	photosDir = "../Dataset/Photos"
	depthsDir = "../Dataset/Depths"
	trainDir = "../Dataset/Train"
	annotations = ""
	styleCount = [0] * 16
	for fileName in os.listdir(photosDir):
		fileDir = os.path.join(photosDir, fileName)
		if (os.path.isdir(fileDir)):
			for styleName in os.listdir(fileDir):
				styleDir = os.path.join(fileDir, styleName)
				if (os.path.isdir(styleDir)):
					for arcName in os.listdir(styleDir):
						arcDir = os.path.join(styleDir, arcName)
						if (os.path.isdir(arcDir)):
							for photoName in os.listdir(arcDir):
								photoDir = os.path.join(arcDir, photoName)
								if (os.path.isfile(photoDir)):
									count += 1
									styleCount[styleNames.index(styleName)] += 1
									if (count % 100 == 0):
										print("processing (%d)-th image..." % (count))
									try:
										photo = cv2.imread(os.path.join(photosDir, fileName, styleName, arcName, photoName))
										depth = cv2.imread(os.path.join(depthsDir, fileName, styleName, arcName, photoName))
										cat = np.concatenate([photo, depth], 1)
										trainName = styleName + str(styleCount[styleNames.index(styleName)]) + ".jpg"
										cv2.imwrite(os.path.join(trainDir, trainName), cat)
										annotations += trainName + ", " + str(styleNames.index(styleName)) + "\n"
									except:
										print("Error in %s %s %s" % (styleName, arcName, photoName))

	annotationFile = open("annotation.csv", "w")
	n = annotationFile.write(annotations)
	annotationFile.close()