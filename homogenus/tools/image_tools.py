# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
# Code Developed by: Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# 2018.11.07

import cv2
import numpy as np
import os

fontColors = {'red': (255, 0, 0),
              'green': (0, 255, 0),
              'yellow': (255, 255, 0),
              'blue': (0, 255, 255),
              'orange': (255, 165, 0),
              'black': (0, 0, 0),
              'grey': (169, 169, 169),
              'white': (255, 255, 255),
              }

def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    cropped = image[offset_height:offset_height + target_height, offset_width:offset_width + target_width, :]
    return cropped

def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    height, width, depth = image.shape

    after_padding_width = target_width - offset_width - width

    after_padding_height = target_height - offset_height - height
    # Do not pad on the depth dimensions.
    paddings = ((offset_height, after_padding_height), (offset_width, after_padding_width), (0, 0))
    padded = np.pad(image, paddings, 'constant')

    return padded

def resize_image_with_crop_or_pad(image, target_height, target_width):
    # crop to ratio, center
    height, width, c = image.shape

    width_diff = target_width - width
    offset_crop_width = max(-width_diff // 2, 0)
    offset_pad_width = max(width_diff // 2, 0)

    height_diff = target_height - height
    offset_crop_height = max(-height_diff // 2, 0)
    offset_pad_height = max(height_diff // 2, 0)

    # Maybe crop if needed.
    # print('image shape', image.shape)
    cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                   min(target_height, height),
                                   min(target_width, width))
    # print('after cropp', cropped.shape)
    # Maybe pad if needed.
    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                  target_height, target_width)
    # print('after pad', resized.shape)
    return resized[:target_height, :target_width, :]

def cropout_openpose(im_fname, pose, want_image=True, crop_margin=0.08):
    im_orig = cv2.imread(im_fname, 3)

    im_height, im_width = im_orig.shape[0], im_orig.shape[1]

    pose = pose[pose[:, 2] > 0.0]

    x_min, x_max = pose[:, 0].min(), pose[:, 0].max()
    y_min, y_max = pose[:, 1].min(), pose[:, 1].max()

    margin_h = crop_margin * im_height
    margin_w = crop_margin * im_width
    offset_height = int(max((y_min - margin_h), 0))
    target_height = int(min((y_max + margin_h), im_height)) - offset_height
    offset_width = int(max((x_min - margin_w), 0))
    target_width = int(min((x_max + margin_w), im_width)) - offset_width

    crop_info = {'crop_boundary':
                     {'offset_height':offset_height,
                     'target_height':target_height,
                     'offset_width':offset_width,
                     'target_width':target_width}}


    if want_image:
        crop_info['cropped_image'] = crop_to_bounding_box(im_orig, offset_height, offset_width, target_height, target_width)

    return crop_info

def put_text_in_image(images, text, color ='white', position=None):
    '''

    :param images: 4D array of images
    :param text: list of text to be printed in each image
    :param color: the color or colors of each text
    :return:
    '''
    import cv2

    if not isinstance(text, list): text = [text]
    if not isinstance(color, list): color = [color for _ in range(images.shape[0])]
    if images.ndim == 3: images = images.reshape(1,images.shape[0],images.shape[1],3)
    images_out = []
    for imIdx in range(images.shape[0]):
        img = images[imIdx].astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if position is None:position = (10, img.shape[1])
        fontScale = 1.
        lineType = 2
        fontColor = fontColors[color[imIdx]]
        cv2.putText(img, text[imIdx],
                    position,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        images_out.append(img)
    return np.array(images_out)

def read_prep_image(im_fname, avoid_distortion=True):
    '''
    if min(height, width) is larger than 224 subsample to 224. this will also affect the larger dimension.
    in the end crop and pad the whole image to get to 224x224
    :param im_fname:
    :return:
    '''
    import cv2

    if isinstance(im_fname, np.ndarray):
        image_data = im_fname
    else:
        image_data = cv2.imread(im_fname, 3)

        # height, width = image_reader.read_image_dims(sess, image_data)
        # image_data = image_reader.decode_jpeg(sess, image_data)

    # print(image_data.min(), image_data.max(), image_data.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(image_data[:,:,::-1].astype(np.uint8))
    # plt.show()

    # height, width = image_data.shape[0], image_data.shape[1]
    # if min(height, width) > 224:
    #     print(image_data.shape)
    #     rt = 224. / min(height, width)
    #     image_data = cv2.resize(image_data, (int(rt * width), int(rt * height)), interpolation=cv2.INTER_AREA)
    #     print('>>resized to>>',image_data.shape)

    height, width = image_data.shape[0], image_data.shape[1]

    if avoid_distortion:
        if max(height, width) > 224:
            # print(image_data.shape)
            rt = 224. / max(height, width)
            image_data = cv2.resize(image_data, (int(rt * width), int(rt * height)), interpolation=cv2.INTER_AREA)
            # print('>>resized to>>',image_data.shape)
    else:

        from skimage.transform import resize

        image_data = resize(image_data, (224, 224), mode='constant', anti_aliasing=False, preserve_range=True)

    # print(image_data.min(), image_data.max(), image_data.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(image_data[:,:,::-1].astype(np.uint8))
    # plt.show()

    image_data = resize_image_with_crop_or_pad(image_data, 224, 224)

    # print(image_data.min(), image_data.max(), image_data.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(image_data[:, :, ::-1].astype(np.uint8))
    # plt.show()

    #return image_data.astype(np.float32)
    return image_data.astype(np.uint8)


def save_images(images, out_dir, im_names = None):
    from homogenus.tools.omni_tools import id_generator

    if images.ndim == 3: images = images.reshape(1,images.shape[0],images.shape[1],3)

    from PIL import Image
    if im_names is None:
        im_names = ['%s.jpg'%id_generator(4) for i in range(images.shape[0])]
    for imIdx in range(images.shape[0]):
        result = Image.fromarray(images[imIdx].astype(np.uint8))
        result.save(os.path.join(out_dir, im_names[imIdx]))
    return True