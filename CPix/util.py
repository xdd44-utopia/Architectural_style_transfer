import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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