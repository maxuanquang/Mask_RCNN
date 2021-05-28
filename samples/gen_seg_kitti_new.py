# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Offline data generation for the KITTI dataset."""

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import cv2
import os, glob


SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1
INPUT_DIR = '/content/raw_data'
OUTPUT_DIR = '/content/drive/MyDrive/Dự án/KITTI Dataset/KITTI Wild'

########################## Align masks #########################################
def compute_overlap(mask1, mask2):
    # Use IoU here.
    return np.sum(mask1 & mask2)/np.sum(mask1 | mask2)

def align(seg_img1, seg_img2, seg_img3, threshold_same=0.3):
    res_img1 = np.zeros_like(seg_img1)
    res_img2 = np.zeros_like(seg_img2)
    res_img3 = np.zeros_like(seg_img3)
    remaining_objects2 = list(np.unique(seg_img2.flatten()))
    remaining_objects3 = list(np.unique(seg_img3.flatten()))
    for seg_id in np.unique(seg_img1):
        # See if we can find correspondences to seg_id in seg_img2.
        max_overlap2 = float('-inf')
        max_segid2 = -1
        for seg_id2 in remaining_objects2:
            overlap = compute_overlap(seg_img1==seg_id, seg_img2==seg_id2)
            if overlap>max_overlap2:
                max_overlap2 = overlap
                max_segid2 = seg_id2
        if max_overlap2 > threshold_same:
            max_overlap3 = float('-inf')
            max_segid3 = -1
            for seg_id3 in remaining_objects3:
                overlap = compute_overlap(seg_img2==max_segid2, seg_img3==seg_id3)
                if overlap>max_overlap3:
                    max_overlap3 = overlap
                    max_segid3 = seg_id3
            if max_overlap3 > threshold_same:
                res_img1[seg_img1==seg_id] = seg_id
                res_img2[seg_img2==max_segid2] = seg_id
                res_img3[seg_img3==max_segid3] = seg_id
                remaining_objects2.remove(max_segid2)
                remaining_objects3.remove(max_segid3)
    return res_img1, res_img2, res_img3
################################################################################

################################################################################
########################## Initialize Segmentation Model #######################
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    DETECTION_MIN_CONFIDENCE = 0.6

config = InferenceConfig()
config.display()

batch_size = 3

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
################################################################################

def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1-middle_perc
    half = left/2
    a = img[int(img.shape[0]*(half)):int(img.shape[0]*(1-half)), :]
    aseg = segimg[int(segimg.shape[0]*(half)):int(segimg.shape[0]*(1-half)), :]
    cy /= (1/middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128*a.shape[1]/a.shape[0]))
    x_scaling = float(wdt)/a.shape[1]
    y_scaling = 128.0/a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx*=x_scaling
    fy*=y_scaling
    cx*=x_scaling
    cy*=y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1]/416)
    c = b[:, int(remain/2):b.shape[1]-int(remain/2)]
    cseg = bseg[:, int(remain/2):b.shape[1]-int(remain/2)]

    return c, cseg, fx, fy, cx, cy


def run_all():
  ct = 0
if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR = OUTPUT_DIR + '/'

for d in glob.glob(INPUT_DIR + '/*/'):
    for d2 in glob.glob(d + '*/'):
        seqname = d2.split('/')[-2]
        print('Processing sequence', seqname)
        for subfolder in ['image_02/data', 'image_03/data']:
            ct = 1
            seqname = d2.split('/')[-2] + subfolder.replace('image', '').replace('/data', '')
            if not os.path.exists(OUTPUT_DIR + seqname):
                os.mkdir(OUTPUT_DIR + seqname)

            folder = d2 + subfolder
            files = glob.glob(folder + '/*.png')
            files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
            files = sorted(files)
            
            # read all images
            batch = []
            for i in range(len(files)):
                img = cv2.imread(files[i])
                batch.append(img)
                
            if len(batch) == 0:
                continue
            elif len(batch) != len(files):
                print("len(batch) != len(files)")
                print(len(batch))
                print(len(files))
                break
            else:
                if len(batch) < batch_size:
                    n = len(batch)
                    add_amount = batch_size - n
                    for _ in range(add_amount):
                        batch.append(batch[-1])
                    results = model.detect(batch, verbose=1)
                    results = results[:n]
                else:
                    n = len(batch)
                    import math
                    nguyen = math.floor(n / batch_size)
                    results = []
                    for i in range(nguyen):
                        results += model.detect(batch[i*batch_size:(i+1)*batch_size], verbose=1)
                    du = n % batch_size
                    if du != 0:
                        add_amount = batch_size - du
                        left_batch = batch[nguyen*batch_size:]
                        for _ in range(add_amount):
                            left_batch.append(left_batch[-1])
                        left_results = model.detect(left_batch, verbose=1)
                        left_results = left_results[:du]
                        results = results + left_results
            
            # Process results
            coarse_masks = []
            for j, r in enumerate(results):
                indices = np.where(r['class_ids'] > 9)[0] 
                indices = indices[::-1]
                if len(indices) > 0:
                    for index in indices:
                        r['masks'] = np.delete(r['masks'], index, axis=2)
                mask = np.zeros(r['masks'].shape)
                for k in range(r['masks'].shape[2]):
                    mask[:,:,k] = r['masks'][:,:,k] * (255-k)
                mask_ = np.sum(mask, 2)
                error = (mask_ <= 255.) * 1.
                mask_ = mask_ * error
                coarse_masks.append(mask_)
            
            print("========================================================")
            print("len(coarse_masks) != len(files)")
            print(len(coarse_masks))
            print(len(files))
            print("========================================================")
            
            # refine results
            for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                print(i)
                imgnum = str(ct).zfill(10)
                if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png'):
                    ct+=1
                    continue
                big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                wct = 0

                # Refine mask list
                res_img1, res_img2, res_img3 = align(coarse_masks[i-SEQ_LENGTH],
                    coarse_masks[i-SEQ_LENGTH+1],
                    coarse_masks[i-SEQ_LENGTH+2], 
                    threshold_same=0.3)

                # Resize and add to big_img
                res_img1 = cv2.resize(res_img1, (WIDTH, HEIGHT))
                res_img2 = cv2.resize(res_img2, (WIDTH, HEIGHT))
                res_img3 = cv2.resize(res_img3, (WIDTH, HEIGHT))

                res_img1 = np.stack((res_img1, res_img1, res_img1), axis=2)
                res_img2 = np.stack((res_img2, res_img2, res_img2), axis=2)
                res_img3 = np.stack((res_img3, res_img3, res_img3), axis=2)

                big_img[:,:WIDTH] = res_img1
                big_img[:,WIDTH:2*WIDTH] = res_img2
                big_img[:,2*WIDTH:] = res_img3

                cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png', big_img)
                ct+=1

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)