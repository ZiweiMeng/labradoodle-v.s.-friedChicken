# code heavily adapted from https://github.com/pengpaiSH/Kaggle_NCFM
import os
import numpy as np
import shutil

np.random.seed(59)

root_train = '../../localData/prj3/training_data/train_split_3d/'
root_val = '../../localData/prj3/training_data/val_split_3d/'
root_test = '../../localData/prj3/training_data/test_split_3d/'

root_total = '../../localData/prj3/training_data/raw_images_3d/'

labels = ['labradoodle', 'friedChicken']

n_train_samples = 0
n_val_samples = 0
n_test_samples = 0

# Training proportion
split_proportion = (0.6,0.2,0.2) # make sure they sum up to 1.0

for label in labels:
    if label not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, label))
        #os.system("sudo mkdir "+os.path.join(root_train, fish))

    total_images = os.listdir(os.path.join(root_total, label))

    n_train = int(len(total_images) * split_proportion[0])
    n_val = int(len(total_images)*split_proportion[1])
    
    np.random.shuffle(total_images)

    train_images = total_images[:n_train]
    val_images = total_images[n_train:(n_train+n_val)]
    test_images = total_images[(n_train+n_val):]

    for img in train_images:
        source = os.path.join(root_total, label, img)
        target = os.path.join(root_train, label, img)
        shutil.copy(source, target)
        n_train_samples += 1

    if label not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, label))

    for img in val_images:
        source = os.path.join(root_total, label, img)
        target = os.path.join(root_val, label, img)
        shutil.copy(source, target)
        n_val_samples += 1
    
    if label not in os.listdir(root_test):
        os.mkdir(os.path.join(root_test, label))

    for img in test_images:
        source = os.path.join(root_total, label, img)
        target = os.path.join(root_test, label, img)
        shutil.copy(source, target)
        n_test_samples += 1

print('Finish splitting train, val and test images!')
print('# training samples: {}, # val samples: {}, # test samples: {}'.format(n_train_samples, n_val_samples, n_test_samples))