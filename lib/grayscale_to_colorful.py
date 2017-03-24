import os
import numpy as np
import shutil
import numpy as np
import cv2
from time import time 

source_total = '../data/test_2d/test/'
#target_total = '../../localData/prj3/training_data/raw_images_3d/'
target_total = '../data/test/test/'

#labels = ['labradoodle', 'friedChicken']

n_total = 0

'''
for label in labels:
    if label not in os.listdir(target_total):
        os.mkdir(os.path.join(target_total, label))

    total_images = os.listdir(os.path.join(source_total, label))

    for img in total_images:
        source = os.path.join(source_total, label, img)
        target = os.path.join(target_total, label, img)
        this_img = cv2.imread(source,cv2.IMREAD_GRAYSCALE)
        this_img = np.dstack((this_img,this_img,this_img))
        cv2.imwrite(target,this_img)
        #shutil.copy(source, target)
        n_total += 1
'''
total_image = os.listdir(source_total)
total_images = []
for x in total_image:
    if x!='.DS_Store':
        total_images.append(x)

print('start reading images...')
s1 = time()
for img in total_images:
        source = os.path.join(source_total, img)
        target = os.path.join(target_total, img)
        this_img = cv2.imread(source,cv2.IMREAD_GRAYSCALE)
        this_img = np.dstack((this_img,this_img,this_img))
        cv2.imwrite(target,this_img)
        #shutil.copy(source, target)
        n_total += 1
s2 = time()





print('Finish convert type of images, uses %.2gs' % (s2 - s1))
print('# Total images converted: {}'.format(n_total))