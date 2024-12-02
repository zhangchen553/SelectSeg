# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 01:14:36 2022

@author: cejize
"""

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import random
from dataclasses import dataclass
from typing import Union
from framework import a_config
from torchvision import transforms  # Image transformation

IMAGE_SIZE = a_config.INPUT_IMAGE_WIDTH
PARAMETER_MAX = 5


@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    shear: MinMax = MinMax(.0, .3)
    translate: MinMax = MinMax(0, int(IMAGE_SIZE / 3))  # different from uniaug: MinMax(0,14.4)
    rotate: MinMax = MinMax(0, 30)
    solarize: MinMax = MinMax(0, 256)
    posterize: MinMax = MinMax(0, 4)  # different from uniaug: MinMax(4,8)
    enhancer: MinMax = MinMax(.1, 1.9)
    cutout: MinMax = MinMax(.0, .2)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def identity(pil_img, pil_mask, _):
    return pil_img, pil_mask


def flip_lr(pil_img, pil_mask, _):
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT), pil_mask.transpose(Image.FLIP_LEFT_RIGHT)


def flip_ud(pil_img, pil_mask, _):
    return pil_img.transpose(Image.FLIP_TOP_BOTTOM), pil_mask.transpose(Image.FLIP_TOP_BOTTOM)


def autocontrast(pil_img, pil_mask, _):
    return ImageOps.autocontrast(pil_img), pil_mask


def equalize(pil_img, pil_mask, _):
    return ImageOps.equalize(pil_img), pil_mask


def blur(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.BLUR), pil_mask


def smooth(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.SMOOTH), pil_mask


def posterize(pil_img, pil_mask, level):
    level = int_parameter(level, min_max_vals.posterize.max - min_max_vals.posterize.min)
    return ImageOps.posterize(pil_img, 4 - level), pil_mask


def rotate(pil_img, pil_mask, level):
    degrees = int_parameter(level, min_max_vals.rotate.max)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR), pil_mask.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, pil_mask, level):
    level = int_parameter(level, min_max_vals.solarize.max)
    return ImageOps.solarize(pil_img, 256 - level), pil_mask


def shear_x(pil_img, pil_mask, level):
    level = float_parameter(level, min_max_vals.shear.max)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR), pil_mask.transform(pil_img.size,
                                                                          Image.AFFINE, (1, level, 0, 0, 1, 0),
                                                                          resample=Image.BILINEAR)


def shear_y(pil_img, pil_mask, level):
    level = float_parameter(level, min_max_vals.shear.max)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR), pil_mask.transform(pil_img.size,
                                                                          Image.AFFINE, (1, 0, 0, level, 1, 0),
                                                                          resample=Image.BILINEAR)


def translate_x(pil_img, pil_mask, level):
    level = int_parameter(level, min_max_vals.translate.max)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR), pil_mask.transform(pil_img.size,
                                                                          Image.AFFINE, (1, 0, level, 0, 1, 0),
                                                                          resample=Image.BILINEAR)


def translate_y(pil_img, pil_mask, level):
    level = int_parameter(level, min_max_vals.translate.max)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR), pil_mask.transform(pil_img.size,
                                                                          Image.AFFINE, (1, 0, 0, 0, 1, level),
                                                                          resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, pil_mask, level):
    level = float_parameter(level, min_max_vals.enhancer.max - min_max_vals.enhancer.min) + min_max_vals.enhancer.min
    return ImageEnhance.Color(pil_img).enhance(level), pil_mask


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, pil_mask, level):
    level = float_parameter(level, min_max_vals.enhancer.max - min_max_vals.enhancer.min) + min_max_vals.enhancer.min
    return ImageEnhance.Contrast(pil_img).enhance(level), pil_mask


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, pil_mask, level):
    level = float_parameter(level, min_max_vals.enhancer.max - min_max_vals.enhancer.min) + min_max_vals.enhancer.min
    return ImageEnhance.Brightness(pil_img).enhance(level), pil_mask


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, pil_mask, level):
    level = float_parameter(level, min_max_vals.enhancer.max - min_max_vals.enhancer.min) + min_max_vals.enhancer.min
    return ImageEnhance.Sharpness(pil_img).enhance(level), pil_mask,


def contour(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.CONTOUR), pil_mask


def detail(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.DETAIL), pil_mask


def edge_enhance(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.EDGE_ENHANCE), pil_mask


def sharpen(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.SHARPEN), pil_mask


def max_(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.MaxFilter), pil_mask


def min_(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.MinFilter), pil_mask


def median_(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.MedianFilter), pil_mask


def gaussian_(pil_img, pil_mask, _):
    return pil_img.filter(ImageFilter.GaussianBlur), pil_mask


min_max_vals = MinMaxVals()

# augmentations = [
#     autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y
# ]

augmentations_trivial = [identity, autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x,
                         translate_y, brightness, contrast, color]  # 加回了一些 augmentations
augmentations_aug = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y]


# augmentations = [ autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,translate_x, translate_y, brightness, contrast, color]

# augmentations = [ autocontrast, equalize, posterize, rotate, solarize, brightness, contrast, color]

# augmentations_all = [
#     autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y, color, contrast, brightness, sharpness
# ]


def apply_op(pil_img, pil_mask, op, level):
    pil_img, pil_mask = op(pil_img, pil_mask, level)
    return pil_img, pil_mask


class TrivialAugment:
    def __call__(self, img, mask):
        img = transforms.ToPILImage()(img)
        mask = transforms.ToPILImage()(mask)
        op = np.random.choice(augmentations_trivial)
        level = random.randint(0, PARAMETER_MAX)
        img, mask = apply_op(img, mask, op, level)
        img = np.array(img)
        mask = np.array(mask)
        return img, mask


class AugMix:
    def __call__(self, img, mask, width=3, depth=-1, alpha=1.):
        ws = np.float32(np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))
        mix_img = np.zeros_like(img).astype(np.float32)
        mix_mask = np.zeros_like(mask).astype(np.float32)

        for i in range(width):
            img_aug = img.copy()
            mask_aug = mask.copy()
            img_aug = transforms.ToPILImage()(img_aug)
            mask_aug = transforms.ToPILImage()(mask_aug)
            d = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = np.random.choice(augmentations_aug)
                level = random.randint(0, PARAMETER_MAX)

                img_aug, mask_aug = apply_op(img_aug, mask_aug, op, level)
            # Preprocessing commutes since all coefficients are convex
            mix_img = mix_img + ws[i] * np.array(img_aug).astype(np.float32)
            mix_mask = mix_mask + ws[i] * np.array(mask_aug).astype(np.float32)

        mixed_img = ((1 - m) * img + m * mix_img).astype(np.uint8)
        mixed_mask = ((1 - m) * mask + m * mix_mask).astype(np.uint8)
        return mixed_img, mixed_mask

        # mixed_img = ((1 - m) * img + m * mix_img).astype(np.float32)/255.0
        # mixed_mask =  ((1 - m) * mask + m * mix_mask).astype(np.float32)
        # if a_config.DATASET != "TOPO":
        #     mixed_mask /= 255.0
        # return mixed_img, mixed_mask


class RandAugment:
    def __call__(self, img, mask):
        n = 2
        m = 9
        img = transforms.ToPILImage()(img)
        mask = transforms.ToPILImage()(mask)
        ops = random.choices(augmentations_trivial, k=n)
        level = int(m / 30 * PARAMETER_MAX)

        for op in ops:
            img, mask = apply_op(img, mask, op, level)
        img = np.array(img)
        mask = np.array(mask)
        return img, mask


class RandAugmentFixMatch:
    def __call__(self, img, mask):
        n = 2
        m = 9
        img = transforms.ToPILImage()(img)
        img_weak = img.copy()
        mask = transforms.ToPILImage()(mask)
        ops = random.choices(augmentations_trivial, k=n)
        level = int(m / 30 * PARAMETER_MAX)

        for op in ops:
            if 'shear' in str(op) or 'translate'  in str(op) or 'rotate' in str(op):
                img, img_weak = apply_op(img, img_weak, op, level)
            else:
                img, mask = apply_op(img, mask, op, level)

        img_strong = np.array(img)
        img_weak = np.array(img_weak)
        mask = np.array(mask)
        return (img_weak, img_strong), mask

class UniAugFixMatch:
    def __call__(self, img, mask):
        img = transforms.ToPILImage()(img)
        img_weak = img.copy()
        mask = transforms.ToPILImage()(mask)

        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)

        img_strong = np.array(img)
        img_weak = np.array(img_weak)
        mask = np.array(mask)
        return (img_weak, img_strong), mask

if __name__ == '__main__':
    # test a_config error because this file in the tool folder, unlesss set semi as false
    img = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    mask = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    # img, mask = TrivialAugment()(img, mask)
    # img, mask = AugMix()(img, mask)
    # img, mask = RandAugment()(img, mask)
    (weak, strong), _ = RandAugmentFixMatch()(img, mask)
    print(img.shape, mask.shape)
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
