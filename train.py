import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from styleNames import styleNames

from CPix.UNet import *
from CPix.PatchGAN import *
from CPix.util import *

adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 100

n_epochs = 500
class_num = 16
input_dim = 3
real_dim = 3
display_step = 2000
checkpoint_step = 20000
batch_size = 1
lr = 0.1
target_shape = 256
device = 'cuda'

transform = transforms.Compose([
	transforms.ToTensor(),
])
dataset = ImagePairDataset("./annotation.csv", "../Dataset/Train")
# gen = UNet(input_dim + class_num, real_dim).to(device)
gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
# disc = Discriminator(input_dim + real_dim + class_num).to(device)
disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def train(save_model=True):
	mean_generator_loss = 0
	mean_discriminator_loss = 0
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	cur_step = 0

	for epoch in range(n_epochs):
		for image, label in tqdm(dataloader):
			image_width = image.shape[3]
			condition = image[:, :, :, :image_width // 2]
			condition = nn.functional.interpolate(condition, size=target_shape)
			real = image[:, :, :, image_width // 2:]
			real = nn.functional.interpolate(real, size=target_shape)
			condition = condition.to(device)
			real = real.to(device)
			
			# oneHotLabel = get_one_hot_labels(label.to('cuda'), 16)
			# imageOneHotLabel = oneHotLabel[:, :, None, None]
			# imageOneHotLabel = imageOneHotLabel.repeat(1, 1, target_shape, target_shape)

			# conditionLabeled = torch.cat((condition.detach(), imageOneHotLabel.float()), 1)
			conditionLabeled = condition

			disc_opt.zero_grad()
			with torch.no_grad():
				fake = gen(conditionLabeled)
			disc_fake_hat = disc(fake.detach(), conditionLabeled)
			disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
			disc_real_hat = disc(real, conditionLabeled)
			disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
			disc_loss = (disc_fake_loss + disc_real_loss) / 2
			disc_loss.backward(retain_graph=True)
			disc_opt.step()

			gen_opt.zero_grad()
			gen_loss = get_gen_loss(gen, disc, real, conditionLabeled, adv_criterion, recon_criterion, lambda_recon)
			gen_loss.backward()
			gen_opt.step()

			mean_discriminator_loss += disc_loss.item() / display_step
			mean_generator_loss += gen_loss.item() / display_step

			if (cur_step % display_step == 0 and cur_step > 0):
				if cur_step > 0:
					print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
				else:
					print("Pretrained initial state")
				mean_generator_loss = 0
				mean_discriminator_loss = 0
				sampleFake = recoverImage(fake[0].to('cpu'))
				sampleReal = recoverImage(real[0].to('cpu'))
				sampleCondition = recoverImage(condition[0].to('cpu'))
				sample = torch.cat((sampleCondition, sampleReal, sampleFake), 2)
				write_jpeg(sample , "./Temp/" + " " + str(cur_step) + ".jpg")
			if (cur_step % checkpoint_step == 0 and save_model):
				torch.save({'gen': gen.state_dict(),
					'gen_opt': gen_opt.state_dict(),
					'disc': disc.state_dict(),
					'disc_opt': disc_opt.state_dict()
				}, f"Checkpoints/pix2pix_{cur_step}.pth")
			cur_step += 1

pretrained = False
if pretrained:
	loaded_state = torch.load("pix2pix_15000.pth")
	gen.load_state_dict(loaded_state["gen"])
	gen_opt.load_state_dict(loaded_state["gen_opt"])
	disc.load_state_dict(loaded_state["disc"])
	disc_opt.load_state_dict(loaded_state["disc_opt"])
else:
	gen = gen.apply(weights_init)
	disc = disc.apply(weights_init)

torch.cuda.empty_cache() 
train()