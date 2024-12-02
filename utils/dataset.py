from torch.utils.data import Dataset
import cv2
from torchvision import transforms  # Image transformation
import utils.aug_lib_new as aug_lib_new
from utils.sda import Rand_Augment
import random

import albumentations as A
from framework import a_config

augmenter_trivial = aug_lib_new.TrivialAugment()
augmenter_augmix = aug_lib_new.AugMix()
augmenter_randaug = aug_lib_new.RandAugment()
augmenter_randaug_fm = aug_lib_new.RandAugmentFixMatch()
augmenter_uniaug_fm = aug_lib_new.UniAugFixMatch()
augmenter_sda = Rand_Augment(Numbers=3, Magnitude=20, max_Magnitude=40)


# from tool.evaluation import *

def read_index(index_path, shuffle=False):
    img_list = []
    # 在这种写法下，path is a txt file that list the path
    # of all image and mask
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(img_list)
    return img_list


def image_mask_transformation(image, mask, img_trans, aug_trans=None, normalize=True,
                              var_min=10, var_max=400):

    # print(image.shape, mask.shape)

    transformed = img_trans(image=image, mask=mask)
    image = transformed["image"]
    mask = transformed["mask"]

    # image, mask still uint 8 in 0, 255 range

    if aug_trans in ['trivial', 'augmix', 'randaug', 'sda']:
        image, mask = eval('augmenter_' + aug_trans)(image, mask)

    if aug_trans == 'sda':
        aux_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, interpolation=1, border_mode=0),
        ])

        transformed = aux_trans(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

    elif aug_trans:
        base_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, interpolation=1, border_mode=0),
        ])
        if aug_trans == 'augmix_gaussian':
            image, mask = eval('augmenter_' + 'augmix')(image, mask)

            aux_trans = A.Compose([
                base_trans,
                A.GaussNoise(p=1.0, var_limit=(var_min, var_max))
            ])

        if aug_trans == 'base':
            aux_trans = base_trans

        if aug_trans == 'gaussianvvar':
            aux_trans = A.Compose([
                base_trans,
                A.GaussNoise(p=1.0, var_limit=(var_min, var_max))
                # still need to confirm the variance range --
                # possibly 0.5**2*255 = 63.5 https://arxiv.org/pdf/2001.06057.pdf
            ])

        transformed = aux_trans(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        if aug_trans == 'fmcolor':
            if random.random() < 0.8:
                image = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image)
                image = transforms.RandomGrayscale(p=0.2)(image)

    # mean, std is calculated in [0, 1] range, it does not matter if current image is in 0-255, or 0-1 range.
    # Reason: just set the max_pixel_value to corresponding range max.
    if normalize:
        img_norm = A.Normalize(a_config.MEAN, a_config.STD, max_pixel_value=255.0, p=1.0)
        transformed = img_norm(image=image)
        image = transformed["image"]
    # After to tensor: the shape of image change from 320*320*3 to 3*320*320
    # If image is in 0-255 range and format in uint8 then change to 0-1, otherwise just keep
    return transforms.ToTensor()(image), transforms.ToTensor()(mask)


def beco_image_mask_transformation(image, mask, confi_mask, img_trans, aug_trans=None, normalize=True):

    # print(image.shape, mask.shape)

    transformed = img_trans(image=image, mask=mask)
    image = transformed["image"]
    mask = transformed["mask"]
    confi_mask = transformed["mask"]

    # image, mask still uint 8 in 0, 255 range

    if aug_trans in ['trivial', 'augmix', 'randaug', 'sda']:
        image, mask, confi_mask = eval('augmenter_' + aug_trans)(image, mask, confi_mask)

    if aug_trans == 'base':
        aux_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, interpolation=1, border_mode=0),
        ])

        transformed = aux_trans(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        confi_mask = transformed["mask"]

    # mean, std is calculated in [0, 1] range, it does not matter if current image is in 0-255, or 0-1 range.
    # Reason: just set the max_pixel_value to corresponding range max.
    if normalize:
        img_norm = A.Normalize(a_config.MEAN, a_config.STD, max_pixel_value=255.0, p=1.0)
        transformed = img_norm(image=image)
        image = transformed["image"]
    # After to tensor: the shape of image change from 320*320*3 to 3*320*320
    # If image is in 0-255 range and format in uint8 then change to 0-1, otherwise just keep
    return transforms.ToTensor()(image), transforms.ToTensor()(mask), transforms.ToTensor()(confi_mask)


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, img_trans, aug_trans=None,
                 normalize=True, var_min=10, var_max=50, fixmatch=False, fm_aug='randaug_fm'):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.img_trans = img_trans
        self.aug_trans = aug_trans
        self.normalize = normalize

        # only for gaussian variant var aug
        self.var_min = var_min
        self.var_max = var_max

        # labeld data or unlabeled data is only for fixmatch or unsup part with different augmentation
        self.fixmatch = fixmatch
        self.fm_aug = fm_aug

    def __len__(self):
        # Number of images
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]

        image = cv2.imread(imagePath)
        # OpenCV loads an image in the BGR format,
        # which we convert to the RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read image as grayscale
        mask = cv2.imread(self.maskPaths[idx], 0)  # mask max 255
        # mask = transforms.ToPILImage()(mask)

        if self.fixmatch:
            transformed = self.img_trans(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            (img_weak, img_strong), _ = eval('augmenter_' + self.fm_aug)(image, mask)

            img_norm = A.Normalize(a_config.MEAN, a_config.STD, max_pixel_value=255.0, p=1.0)
            transformed_weak = img_norm(image=img_weak)
            img_weak = transformed_weak["image"]
            transformed_strong = img_norm(image=img_strong)
            img_strong = transformed_strong["image"]

            return transforms.ToTensor()(img_weak), transforms.ToTensor()(img_strong)

        else:
            image_store, mask_store = image_mask_transformation(image, mask, self.img_trans, aug_trans=self.aug_trans,
                                                                normalize=self.normalize,
                                                                var_min=self.var_min, var_max=self.var_max)
            # mask is already in 0-1 range
            return image_store, mask_store


class BECOSegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, confi_maskPath, img_trans, aug_trans=None,
                 normalize=True):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.confi_maskPath = confi_maskPath
        self.img_trans = img_trans
        self.aug_trans = aug_trans
        self.normalize = normalize

    def __len__(self):
        # Number of images
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]

        image = cv2.imread(imagePath)
        # OpenCV loads an image in the BGR format,
        # which we convert to the RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read image as grayscale
        mask = cv2.imread(self.maskPaths[idx], 0)  # mask max 255
        confi_mask = cv2.imread(self.confi_maskPath[idx], 0)  # mask max 255

        image_store, mask_store, confi_mask_store = beco_image_mask_transformation(image, mask, confi_mask,
                                                                                   self.img_trans,
                                                                                   aug_trans=self.aug_trans,
                                                                                   normalize=self.normalize)
        # mask is already in 0-1 range
        return image_store, mask_store, confi_mask_store


def image_mask_transformation_soft(image, mask, soft_mask, img_trans, aug_trans=None, normalize=True,
                                   var_min=10, var_max=400):

    # print(image.shape, mask.shape)

    transformed = img_trans(image=image, mask=mask, soft_mask=soft_mask)
    image = transformed["image"]
    mask = transformed["mask"]
    soft_mask = transformed["mask"]

    # image, mask still uint 8 in 0, 255 range

    if aug_trans in ['trivial', 'augmix', 'randaug']:
        image, mask, soft_mask = eval('augmenter_' + aug_trans)(image, mask, soft_mask)

    elif aug_trans:
        base_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, interpolation=1, border_mode=0),
        ])

        if aug_trans == 'base':
            aux_trans = base_trans

        transformed = aux_trans(image=image, mask=mask, soft_mask=soft_mask)
        image = transformed["image"]
        mask = transformed["mask"]
        soft_mask = transformed["mask"]

    # mean, std is calculated in [0, 1] range, it does not matter if current image is in 0-255, or 0-1 range.
    # Reason: just set the max_pixel_value to corresponding range max.
    if normalize:
        img_norm = A.Normalize(a_config.MEAN, a_config.STD, max_pixel_value=255.0, p=1.0)
        transformed = img_norm(image=image)
        image = transformed["image"]
    # After to tensor: the shape of image change from 320*320*3 to 3*320*320
    # If image is in 0-255 range and format in uint8 then change to 0-1, otherwise just keep
    return transforms.ToTensor()(image), transforms.ToTensor()(mask), transforms.ToTensor()(soft_mask)


class SegmentationDataset_soft(Dataset):
    def __init__(self, imagePaths, maskPaths, softmaskPaths, img_trans, aug_trans=None,
                 normalize=True, var_min=10, var_max=50):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.softmaskPaths = softmaskPaths
        self.img_trans = img_trans
        self.aug_trans = aug_trans
        self.normalize = normalize

        # only for gaussian variant var aug
        self.var_min = var_min
        self.var_max = var_max

    def __len__(self):
        # Number of images
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]

        image = cv2.imread(imagePath)
        # OpenCV loads an image in the BGR format,
        # which we convert to the RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read image as grayscale
        mask = cv2.imread(self.maskPaths[idx], 0)  # mask max 255
        soft_mask = cv2.imread(self.softmaskPaths[idx], 0)  # mask max 255
        # mask = transforms.ToPILImage()(mask)

        image_store, mask_store, soft_mask = image_mask_transformation_soft(image, mask, soft_mask, self.img_trans,
                                                                            aug_trans=self.aug_trans,
                                                                            normalize=self.normalize,
                                                                            var_min=self.var_min, var_max=self.var_max)
        # mask is already in 0-1 range
        return image_store, mask_store, soft_mask

class NpyDataset(Dataset):
    """
    Read .npy files directly
    """

    def __init__(self,
                 data,
                 targets,
                 num_classes=None,
                 img_trans=None,
                 aug_trans=False,
                 var_min=10,
                 var_max=50,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(NpyDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes

        self.img_trans = img_trans
        self.aug_trans = aug_trans

        # only for gaussian variant var aug
        self.var_min = var_min
        self.var_max = var_max

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        mask = self.targets[idx]
        # set augmented images
        image = self.data[idx]

        image, mask = image_mask_transformation(image, mask, self.img_trans, self.aug_trans,
                                                var_min=self.var_min, var_max=self.var_max)
        return image, mask  # 去掉了idx输出，evaluate_C 里的也应该对应去掉

    def __len__(self):
        return len(self.data)


class FFTDataset(Dataset):

    def __init__(self,
                 data,
                 targets,
                 num_classes=None,
                 img_trans=None,
                 aug_trans=False,
                 var_min=10,
                 var_max=50,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(FFTDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes

        self.img_trans = img_trans
        self.aug_trans = aug_trans

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        mask = self.targets[idx]
        # set augmented images
        image = self.data[idx]

        image, mask = image_mask_transformation(image, mask, self.img_trans, self.aug_trans, )
        return image, mask

    def __len__(self):
        return len(self.data)
