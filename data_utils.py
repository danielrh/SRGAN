from os import listdir, path
import os
from os.path import join
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torchvision.transforms.functional as TF

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        CenterCrop(488),
        ToTensor()
    ])

def replace_hi_lo(filename):
    ret = filename.replace('-hi.png', '-lo.png').replace('-hs.png', '-ls.png')
    return ret
    #dirname = os.path.dirname(ret)
    #basename = os.path.basename(ret)
    #return os.path.join(dirname, "..", "DIV2K_train_LR", basename)

class TrainDatasetFromFolder(Dataset):
    def __init__(
            self,
            dataset_dir,
            crop_size,
            upscale_factor,
            cur_epoch,
            advance_upscale_at_epoch,
    ):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x) and not x.endswith("-lo.png") and not x.endswith("-md.png") and not x.endswith("-ls.png") and not x.endswith("-ms.png")]
        self.upscale_factor = upscale_factor
        self.cur_epoch = cur_epoch
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)
        self.advance_upscale_at_epoch = advance_upscale_at_epoch

    def __getitem__(self, index):
        hr_precrop = Image.open(self.image_filenames[index]).convert('RGB')
        if (self.image_filenames[index].endswith('-hi.png') and self.cur_epoch[0] >= self.advance_upscale_at_epoch) or self.image_filenames[index].endswith('-hs.png'):
            lr_precrop = Image.open(replace_hi_lo(self.image_filenames[index])).convert('RGB')
        else:
            lr_precrop = Resize(tuple(x//self.upscale_factor for x in hr_precrop.size), interpolation=Image.BICUBIC)(hr_precrop)
        crop_indices = RandomCrop.get_params(
                lr_precrop, output_size=(self.crop_size // self.upscale_factor, self.crop_size//self.upscale_factor))
        hr_crop_indices = (crop_indices[0] * self.upscale_factor,
                           crop_indices[1] * self.upscale_factor,
                           crop_indices[2] * self.upscale_factor,
                           crop_indices[3] * self.upscale_factor)
        hr_cropped = TF.crop(hr_precrop, hr_crop_indices[0],hr_crop_indices[1],hr_crop_indices[2],hr_crop_indices[3])
        lr_cropped = TF.crop(lr_precrop, crop_indices[0],crop_indices[1],crop_indices[2],crop_indices[3])
        
        #hr_cropped.save("/tmp/hr" + str(hr_crop_indices[0])+"."+str(hr_crop_indices[1])+"."+str(hr_crop_indices[2])+"." + str(hr_crop_indices[3])+"."+path.basename(self.image_filenames[index]).replace(" ", "-"), "PNG")
        #lr_cropped.save("/tmp/lr" + str(crop_indices[0])+"."+str(crop_indices[1])+"."+str(crop_indices[2])+"." + str(crop_indices[3])+"."+ path.basename(self.image_filenames[index]).replace(" ", "-"), "PNG")
        hr_image = ToTensor()(hr_cropped)
        lr_image = ToTensor()(lr_cropped)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x) and not x.endswith("-lo.png") and not x.endswith("-md.png") and not x.endswith("-ls.png") and not x.endswith("-ms.png")]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        if self.image_filenames[index].endswith("-hi.png") or self.image_filenames[index].endswith('-hs.png'):
            lr_image = CenterCrop(crop_size // self.upscale_factor)(Image.open(replace_hi_lo(self.image_filenames[index])).convert('RGB'))
            hr_restore_img = hr_scale(lr_image)
        else:
            lr_image = lr_scale(hr_image)
            hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index]).convert('RGB')
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index]).convert('RGB')
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
