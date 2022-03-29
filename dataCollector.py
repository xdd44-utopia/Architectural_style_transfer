from bing_images import bing
from styleNames import styleNames

for style, architectures in styleNames.items():
	for architecture in architectures:
		print(style, ' ', architecture)
		bing.download_images(architecture,
							1000,
							output_dir="/Users/xdd44/Dropbox/Final/Dataset/" + style + "/" + architecture,
							pool_size=10,
							file_type="",
							force_replace=True)

# export PATH=$PATH:/Users/xdd44/Dropbox/Final