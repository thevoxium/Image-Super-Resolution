import os
from os import listdir
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image, ImageFilter
from os.path import join

CROP_SIZE = 32

def is_image_file(filename):
    #any() returns true if one of the thing is true in list
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    #converting to ycbcr and getting the Y band after spliting
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(Dataset):
    #this class will inherit pytorch dataset
    def __init__(self, image_dir, zoom_factor):
        super(DatasetFromFolder).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        crop_size = CROP_SIZE - (CROP_SIZE % zoom_factor)
        self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                transforms.Resize(crop_size//zoom_factor),
                                transforms.Resize(crop_size, interpolation=Image.BICUBIC),
                                transforms.ToTensor()])

        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                transforms.ToTensor()])

    #getitem supports indexing of the Dataset
    #length returns size of the dataset
    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
