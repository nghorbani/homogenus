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


from PIL import Image  # $ sudo apt-get install python-imaging
import numpy as np
import os
import json
from homogenus.tools.image_tools import cropout_openpose

def should_accept_pose(pose, human_prob_thr=.5):
    '''

    :param pose:
    :param human_prob_thr:
    :return:
    '''
    rleg_ids = [12,13]
    lleg_ids = [9,10]
    rarm_ids = [5,6,7]
    larm_ids = [2,3,4]
    head_ids = [16,15,14,17,0,1]

    human_prob = pose[:, 2].mean()

    rleg = sum(pose[rleg_ids][:, 2] > 0.0)
    lleg = sum(pose[lleg_ids][:, 2] > 0.0)
    rarm = sum(pose[rarm_ids][:, 2] > 0.0)
    larm = sum(pose[larm_ids][:, 2] > 0.0)
    head = sum(pose[head_ids][:, 2] > 0.0)

    if rleg<1 and lleg<1: return False
    if rarm<1 and larm<1: return False
    if head<2: return False

    if human_prob < human_prob_thr: return False
    return True

def crop_humans(im_fname, pose_fname, want_image=True, human_prob_thr=0.5):
    '''

    :param im_fname: the input image path
    :param pose_fname: the corresponding openpose json file
    :param want_image: if False will only return the crop boundaries, otherwise will also return the cropped image
    :param human_prob_thr: the probability to threshold the detected humans
    :return:
    '''
    crop_infos = {}


    with open(pose_fname) as f: pose_data = json.load(f)

    if not len(pose_data['people']): return crop_infos
    for pIdx in range(len(pose_data['people'])):
        pose = np.asarray(pose_data['people'][pIdx]['pose_keypoints_2d']).reshape(-1, 3)
        if not should_accept_pose(pose, human_prob_thr=human_prob_thr): continue

        crop_info = cropout_openpose(im_fname, pose, want_image=want_image)

        crop_infos['%02d'%pIdx] = {'crop_info':crop_info['crop_boundary'],
                                   #'pose':pose,
                                   'pose_hash':hash(pose.tostring())}
        if want_image:
            crop_infos['%02d' % pIdx]['cropped_image'] = crop_info['cropped_image'].astype(np.uint8)

    return crop_infos

def crop_dataset(base_dir, dataset_name, human_prob_thr=.5):
    import glob
    import random

    results_dir = os.path.join(base_dir, dataset_name, 'cropped_body_tight')
    images_dir = os.path.join(base_dir, dataset_name, 'images')
    pose_jsonpath = os.path.join(base_dir, dataset_name, 'openpose_json')

    if not os.path.exists(results_dir): os.makedirs(results_dir)
    crop_infos_jsonpath = os.path.join(results_dir, 'crop_infos.json')

    fnames = glob.glob(os.path.join(images_dir, '*.jpg'))
    random.shuffle(fnames)

    crop_infos = {}

    for fname in fnames:

        pose_fname = os.path.join(pose_jsonpath, os.path.basename(fname).replace('.jpg', '_keypoints.json'))
        if not os.path.exists(pose_fname): continue

        cur_crop_info = crop_humans(fname, pose_fname, human_prob_thr=human_prob_thr, want_image=True)

        for pname in cur_crop_info.keys():

            crop_id = '%s_%s' % (os.path.basename(fname).split('.')[0], pname)
            crop_outpath = os.path.join(results_dir, '%s.jpg' % crop_id)

            cropped_image = cur_crop_info[pname]['cropped_image']
            if cropped_image.shape[0] < 200 or cropped_image.shape[1] < 200: continue

            cur_crop_info.pop('cropped_image')
            # print(crop_outpath)

            result = Image.fromarray(cropped_image[:,:,::-1])
            result.save(crop_outpath)
            crop_infos[crop_id] = cur_crop_info

    with open(crop_infos_jsonpath, 'w') as f:
        json.dump(crop_infos, f)