from torch import imag
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
import pandas as pd
from CPix.util import *

dataset = ImagePairDataset("annotation.csv", "../Dataset/Train")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
count = 0
for photo, label in dataloader:
	count += 1
	width = photo.shape[3]
	print(photo.shape, label)
	real = photo[:, :, :, :width // 2]
	condition = photo[:, :, :, width // 2:]

	write_jpeg(recoverImage(condition[0].to('cpu')), "./Temp/condition" + str(count) + ".jpg")
	write_jpeg(recoverImage(real[0].to('cpu')), "./Temp/real" + str(count) + ".jpg")

	oneHotLabel = get_one_hot_labels(label.to('cuda'), 16)
	imageOneHotLabel = oneHotLabel[:, :, None, None]
	imageOneHotLabel = imageOneHotLabel.repeat(1, 1, 512, 512)

	print(imageOneHotLabel.shape)
	conditionLabeled = torch.cat((condition.detach(), imageOneHotLabel.float()), 1)
	realLabeled = torch.cat((real.detach(), imageOneHotLabel.float()), 1)

	write_jpeg(condition.to('cpu'), "./Temp/condition" + str(count) + ".jpg")
	write_jpeg(real.to('cpu'), "./Temp/real" + str(count) + ".jpg")
	if (count == 10):
		break