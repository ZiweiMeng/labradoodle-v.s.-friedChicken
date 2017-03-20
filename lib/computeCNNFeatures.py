from keras.models import load_model, Model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from time import time
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 32
n_samples = 2000
n_augmentation = 5

root_path = './'
weights_path = os.path.join(root_path, '../../localData/prj3/weights.h5')
data_dir = os.path.join(root_path,'../../localData/prj3/training_data/raw_images_3d/')

start_train = time()
print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)
InceptionV3_model = Model(InceptionV3_model.input, InceptionV3_model.layers[-2].output)

datagen = ImageDataGenerator()

data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False, # Important !!!
            seed = 59,
            classes = None,
            class_mode = None)

image_list = data_generator.filenames

features = InceptionV3_model.predict_generator(data_generator, n_samples)
end_train = time()

features = pd.DataFrame(features).transpose()
features.columns = image_list
features.to_csv('../data/cnn_features.csv',index=False)



print('computing CNN features runs for %.4g s' % (end_train-start_train))


