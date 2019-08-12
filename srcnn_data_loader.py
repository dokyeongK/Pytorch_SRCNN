import os
import sys
import tarfile
import collections
import torch.utils.data as data
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch
from PIL import ImageFilter as IF
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class SRCNN(data.DataLoader):

    def __init__(self, root_dir, transform=None, upscale = 0):
        self.transform = transform
        dir = root_dir
        self.upscale_factor = upscale
        file_list_image = [file for file in os.listdir(dir) if file.endswith(".jpg")]
        self.images = [os.path.join(dir, x) for x in file_list_image]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')

        crop = transforms.CenterCrop(
            (int(img.size[1] / self.upscale_factor) * self.upscale_factor,
             int(img.size[0] / self.upscale_factor) * self.upscale_factor))

        img = crop(img)
        out = img.filter(IF.GaussianBlur(1.3))  # .convert('YCbCr') 3채널 : RGB
        out = out.resize((int(out.size[0] / self.upscale_factor), int(out.size[1] / self.upscale_factor)))
        out = out.resize((int(out.size[0] * self.upscale_factor), int(out.size[1] * self.upscale_factor)))

        return self.transform(out), self.transform(img)

    def __len__(self):

        return len(self.images)

