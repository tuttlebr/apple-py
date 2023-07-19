from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
opt = parser.parse_args()

print(opt)
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if not opt.mps and torch.backends.mps.is_available():
    raise Exception("Found mps device, please run with --mps to enable macOS GPU")

use_mps = opt.mps and torch.backends.mps.is_available()

if opt.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

img = Image.open(opt.input_image).convert('YCbCr')
y, cb, cr = img.split()

model = torch.load(opt.model)
img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

if opt.cuda:
    model = model.cuda()
    input = input.cuda()
elif use_mps:
    model = model.to(device)
    input = input.to(device)

out = model(input)
out = out.cpu()
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
