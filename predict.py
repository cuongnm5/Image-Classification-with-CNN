import numpy as np
from keras.preprocessing import image
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import cv2

new_model = tf.keras.models.load_model('my_model.h5')
new_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# new_model.summary()

dirs = os.listdir('data/data_private')
file = open("solve.csv", "a")

for files in dirs:
    file_name = "data/data_private/" + files
    img = cv2.imread(file_name)
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])
    classes = new_model.predict_classes(img)
    x = int(classes)
    file.write(files)
    file.write(",")
    file.write(str(x))
    file.write("\n")
    # print (x)

file.close()

# test_datagen = ImageDataGenerator(rescale = 1./255)
# test_set = test_datagen.flow_from_directory('data/public_test',
# target_size = (64, 64),
# batch_size = 32,
# class_mode = 'categorical')

# out_csv = 'new-predictions.csv'

# predictions = new_model.predict(images)

# prediction = pd.DataFrame(predictions, columns=['Label']).to_csv('prediction.csv')