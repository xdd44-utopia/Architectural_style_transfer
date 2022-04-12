from PIL import Image
import os

image = Image.open("/Users/xdd44/Documents/CS4514/Demo/Depth/depth.jpg")

w, h = image.size

print(h, w)

dir = "/Users/xdd44/Dropbox/Final/Architectural_style_transfer/static/image/Programming/StyleDemo/Results/"

for imageName in os.listdir(dir):
	imageDir = dir + imageName
	print(imageDir)
	if (os.path.isfile(imageDir)):
		result = Image.open(imageDir)
		result = result.resize((w, h))
		result.save(imageDir)