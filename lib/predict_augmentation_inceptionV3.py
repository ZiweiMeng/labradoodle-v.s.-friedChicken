# code heavily adapted from https://github.com/pengpaiSH/Kaggle_NCFM
# config keras to be tensorflow-backened
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 32
n_test_samples = 400
n_augmentation = 5

labels = ['labradoodle', 'friedChicken']

root_path = './'
weights_path = os.path.join(root_path, '../../localData/prj3/weights.h5')
test_data_dir = os.path.join(root_path,'../../localData/prj3/training_data/test_split_3d/')

# test data generator for prediction
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

for idx in range(n_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False, # Important !!!
            seed = random_seed,
            classes = None,
            class_mode = None)

    test_image_list = test_generator.filenames
    #print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        predictions = InceptionV3_model.predict_generator(test_generator, n_test_samples)
    else:
        predictions += InceptionV3_model.predict_generator(test_generator, n_test_samples)

predictions /= n_augmentation

print('Begin to write prediction file using InceptionV3 model..')
f_submit = open(os.path.join(root_path, '../output/prediction_inceptionV3.csv'), 'w')
f_submit.write('image,labradoodle,friedChicken\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, n_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Prediction file successfully generated!')