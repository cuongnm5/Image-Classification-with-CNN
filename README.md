# Image-Classification-with-CNN

Keras - Convolutional Neural Networks - Image Classification

### Phân loại ảnh sử dụng mạng nơ-ron tích chập ( CNN ).

Bài toán đặt ra là phân loại biến báo giao thông thành 8 nhóm cho trước. Bộ dữ liệu được dùng để huấn luyện là lí tưởng ( không có nhiễu ). Mình sử dụng mạng thần kinh 6 lớp để thực hiện điều này, với sự hỗ trợ từ Keras.
Thông tin tham khảo về CNN: [Convolutional Neural Networks](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)

**1. Load dữ liệu**

  Ở đây mình sử dụng ImageDataGenerator, thư viện này hỗ trợ rất tốt việc đọc ảnh, các tính năng cơ bản gồm có định dạng size, xoay ảnh, lật ảnh, phóng to, giúp chúng ta có nhiều dữ liệu hơn phục vụ cho việc huấn luyện. Chi tiết các options các bạn có thể xem ở đây: [ImageDataGenerator](https://keras.io/preprocessing/image/)
  
  ``` python3
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data/public_test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')
```

**2. Xây dựng model**

Vì đây là một bài toán đơn giản với dữ liệu đẹp, model của mình không có quá nhiều layer và các layer không phức tạp.

``` python3 
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second axpooling
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size= (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 8, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
```

**3. Huấn luyện**

Huấn luyện model và lưu lại trọng số vào file .h5

``` python3
#Train model
classifier.fit_generator(training_set,
steps_per_epoch = 6589,
epochs = 5,
validation_data = test_set,
validation_steps = 20)

classifier.save('my_model.h5')
```

**4. Nhận dạng ảnh**

Ảnh cần phân loại ở file 'data/data_privare'. Chúng ta sẽ đọc file đó, lấy ra từng ảnh và sử dụng hàm predict_classes() để phân loại. Ảnh và nhãn đã được phân loại sẽ được lưu vào file 'solve.csv' dưới định dạng <ImageID,Label> 

``` python3

new_model = tf.keras.models.load_model('my_model.h5')
new_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

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

file.close()
```
