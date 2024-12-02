from framework import a_config
import torch.nn as nn
import albumentations as A
import random
import cv2
import segmentation_models_pytorch as smp


# Define a function to initialize the weights of a specific layer
def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


def get_model(backbone_name, decoder_name, dataset_name, prefix='', lr=a_config.INIT_LR,
              lossfn=a_config.LOSSFN, h=a_config.INPUT_IMAGE_HEIGHT, w=a_config.INPUT_IMAGE_WIDTH,
              portion=100, seed=0, aug_type='base', semi=False, cps=False):

    model_name = 'nothing'
    result_name = 'nothing'

    common_name = prefix + dataset_name + '_' \
                  + decoder_name + '_' + backbone_name + '_' + 'Aug'
    if portion:
        common_name = str(int(portion)) + 'pct_' + common_name
    if semi:
        common_name = 'semi_' + common_name

    # Add is_check_shapes=False to avoid error for CRACK500 dataset
    if a_config.platform == 'linux':
        img_trans = A.Compose([A.Resize(h, w,
                                        interpolation=cv2.INTER_NEAREST)], is_check_shapes=False)
    else:
        img_trans = A.Compose([A.Resize(h, w,
                                        interpolation=cv2.INTER_NEAREST)], is_check_shapes=False)

    if aug_type in ['base', 'randaug', 'augmix', 'trivial', 'sda']:
        name = common_name \
               + aug_type + '_LR' + '{:.0E}'.format(lr) + '_LOSS' + lossfn + '_SEED' \
               + str(seed)
        model_name = name + '.pth'
        result_name = name + '.npy'

    elif aug_type == 'gaussian':
        name = common_name + 'gaussianvvar' \
               + str(float(10)) + '_' + str(float(400)) + '_LR' + '{:.0E}'.format(lr) \
               + '_LOSS' + lossfn + '_SEED' + str(seed)
        model_name = name + '.pth'
        result_name = name + '.npy'

    elif aug_type == 'clahe':
        name = 'clahe' + common_name \
               + 'base' + '_LR' + '{:.0E}'.format(lr) + '_LOSS' + lossfn + '_SEED' \
               + str(seed)
        model_name = name + '.pth'
        result_name = name + '.npy'
        img_trans = A.Compose([A.Equalize(mode="pil", p=1),
                               A.Resize(h, w)])

    elif aug_type == 'mix':
        name = 'clahe' + common_name \
               + 'augmix_gaussian' + '_LR' + '{:.0E}'.format(lr) + '_LOSS' + lossfn + '_SEED' \
               + str(seed)
        model_name = name + '.pth'
        result_name = name + '.npy'
        img_trans = A.Compose([A.Equalize(mode="pil", p=1),
                               A.Resize(h, w)])

    encoder_depth = 5
    if 'convnext' in backbone_name:
        encoder_depth = 4

    model = eval('smp.' + decoder_name)(encoder_name=backbone_name,
                                        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                        encoder_weights='imagenet',
                                        # use `imagenet` pre-trained weights for encoder initialization
                                        in_channels=3,
                                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                        classes=a_config.NUM_CLASSES,
                                        # model output channels (number of classes in your dataset))
                                        encoder_depth=encoder_depth,
                                        decoder_merge_policy='cat', decoder_dropout=0.2
                                        )

    if cps:
        model.apply(lambda x: random.seed(seed))
        model.decoder.apply(init_weights)

    return model, model_name, result_name, img_trans


if __name__ == '__main__':
    m, m_name, r_name, trans = get_model('mit_b2', 'FPN', 'crack500', seed=0)

    for module in m.decoder.modules():
        print(module)
        print(isinstance(module, nn.Conv2d), '\n\n')
