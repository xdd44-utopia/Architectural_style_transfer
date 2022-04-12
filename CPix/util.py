import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image

def get_one_hot_labels(labels, n_classes):
    return nn.functional.one_hot(labels, n_classes)

def combine_vectors(x, y):
    return torch.cat((x.float(), y.float()), 1)

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	if isinstance(m, nn.BatchNorm2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
		torch.nn.init.constant_(m.bias, 0)

def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
	fake = gen(condition)
	disc_fake = disc(fake, condition)
	adv_loss = torch.sum(adv_criterion(disc_fake, torch.ones_like(disc_fake)))
	recon_loss = lambda_recon * recon_criterion(fake, real)
	gen_loss = adv_loss + recon_loss
	return gen_loss

def recoverImage(image):
	recoverTransform = transforms.Compose([
		transforms.Normalize(
			mean = [0., 0., 0.],
			std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
		),
		transforms.Normalize(
			mean = [-0.485, -0.456, -0.406],
			std = [1., 1., 1.]),
		])
	return recoverTransform(image).to(torch.uint8)

class ImagePairDataset(Dataset):
	def __init__(self, annotations_file, img_dir):
		self.img_labels = pd.read_csv(annotations_file)
		self.img_dir = img_dir

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		image = read_image(img_path).float()
		label = self.img_labels.iloc[idx, 1]
		transform = transforms.Compose([
			transforms.Resize([512, 1024], transforms.InterpolationMode.BICUBIC),
			transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
		])
		image = transform(image)
		return image.to('cuda'), label