import torchgeometry as tgm
import torch
import matplotlib.pyplot as plt


img = plt.imread('textures\cat.jpg')

torch_img = torch.Tensor(img).reshape((1, 3, img.shape[0], img.shape[1]))
torch_img_blur = tgm.image.gaussian_blur(torch_img, (15, 15), (3, 3))

from PIL import Image
import numpy as np

arr = np.array(torch_img_blur).reshape(torch_img_blur.shape[2], torch_img_blur.shape[3], 3).astype('uint8')
print(arr.shape)
img = Image.fromarray(arr[:, :, 0], mode='P')
print(img)
