import os
import numpy as np
import shutil
import numpy as np
import cv2

source_total = '../../localData/prj3/training_data/raw_images/'
target_total = '../../localData/prj3/training_data/raw_images_3d/'

labels = ['labradoodle', 'friedChicken']

n_total = 0

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

print('Finish convert type of images!')
print('# Total images converted: {}'.format(n_total))