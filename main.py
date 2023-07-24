# Importing required modules/libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import ResNet50V2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPool2D, GlobalAvgPool2D, Add
from tensorflow.keras import Model
from PIL import Image


# Setting filepaths
path_to_labels = 'labelnames.csv'
path_to_training_imgs = 'train'
path_to_testing_imgs = 'test'
path_to_validation_imgs = 'validation'

# Nike
nike_train_img_1 = [
    path_to_training_imgs + '/nike/' + file for file in os.listdir(path_to_training_imgs + '/nike')
]


nike_test_img_1 = [
    path_to_testing_imgs + '/nike/' + file for file in os.listdir(path_to_testing_imgs + '/nike')
]


nike_train_img_list = (nike_train_img_1 
                       
                       + nike_test_img_1
                       )

# Adidas 
adidas_train_img_1 = [
    path_to_training_imgs + '/adidas/' + file for file in os.listdir(path_to_training_imgs + '/adidas')
]


adidas_test_img_1 = [
    path_to_testing_imgs + '/adidas/' + file for file in os.listdir(path_to_testing_imgs + '/adidas')
]


adidas_train_img_list = (adidas_train_img_1 
                       
                       + adidas_test_img_1
                       )

# Create new data frames
nike_filepath_df = pd.DataFrame({
    'file_paths' : nike_train_img_list,
    'label' : 'Nike'
})

adidas_filepath_df = pd.DataFrame({
    'file_paths' : adidas_train_img_list,
    'label' : 'Adidas'
})

img_filepaths_df = pd.concat([adidas_filepath_df, nike_filepath_df], ignore_index=True)

# Reads image from filepath and returns image shape
def getImageShape(path_to_images):
    image = Image.open(path_to_images)
    shape = image.size[::-1]  # Reversing the size to (height, width)
    return shape

# Get image shapes 
image_shapes_list = []
for img in img_filepaths_df['file_paths']:
    image_shapes_list.append(getImageShape(img))

# Create training, validation, and testing subsets using functions
def create_subsets(df, train_frac=0.7, val_frac=0.3):
    total_size = len(df)
    train_size = int(total_size * train_frac)
    val_size = int(train_size * val_frac)
    
    df = df.sample(frac=1, random_state=10)  # Shuffle the dataframe
    
    training_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    testing_df = df.iloc[train_size + val_size:]
    
    return training_df, val_df, testing_df

training_df, val_df, testing_df = create_subsets(img_filepaths_df)

# Creating the Conv-Batch Norm block
def conv_bn(x, filters, kernel_size, strides=1):
    
    x = Conv2D(filters=filters, 
               kernel_size = kernel_size, 
               strides=strides, 
               padding = 'same', 
               use_bias = False)(x)
    x = BatchNormalization()(x)
    return x

# Creating separableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
    
    x = SeparableConv2D(filters=filters, 
                        kernel_size = kernel_size, 
                        strides=strides, 
                        padding = 'same', 
                        use_bias = False)(x)
    x = BatchNormalization()(x)
    return x

# Model code...
def entry_flow(x):
    
    x = conv_bn(x, filters =32, kernel_size =3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters =64, kernel_size =3, strides=1)
    tensor = ReLU()(x)
    
    x = sep_bn(tensor, filters = 128, kernel_size =3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 128, kernel_size =3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=128, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=256, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=728, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    return x

# Middle flow

def middle_flow(tensor):
    
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
        
    return tensor

# Exit flow

def exit_flow(tensor):
    
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 1024,  kernel_size=3)
    x = MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
    
    tensor = conv_bn(tensor, filters =1024, kernel_size=1, strides =2)
    x = Add()([tensor,x])
    
    x = sep_bn(x, filters = 1536,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 2048,  kernel_size=3)
    x = GlobalAvgPool2D()(x)
    
    x = Dense (units = 1, activation = 'sigmoid')(x)
    
    return x

# Model definition...
input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)
x = entry_flow(inputs)
x = middle_flow(x)
outputs = exit_flow(x)

model = Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define batch sizes
batch_size = 150
val_batch_size = 70

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    horizontal_flip=True,
    channel_shift_range=True
)

# Data augmentation for validation and testing sets
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    horizontal_flip=True,
    channel_shift_range=True
)

# Normalization for testing set
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(training_df,
                                                    x_col='file_paths',
                                                    y_col='label',
                                                    class_mode='binary',
                                                    batch_size=batch_size,
                                                    target_size=(224, 224),
                                                    shuffle=True,
                                                    seed=10)

validation_generator = val_datagen.flow_from_dataframe(val_df,
                                                       x_col='file_paths',
                                                       y_col='label',
                                                       class_mode='binary',
                                                       batch_size=val_batch_size,
                                                       target_size=(224, 224),
                                                       shuffle=True,
                                                       seed=11)

test_generator = test_datagen.flow_from_dataframe(testing_df,
                                                  x_col='file_paths',
                                                  y_col='label',
                                                  class_mode='binary',
                                                  batch_size=val_batch_size,
                                                  target_size=(224, 224),
                                                  shuffle=False,
                                                  seed=13)
# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto", restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

# Training
epochs = 30
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    verbose=2,
    callbacks=callbacks
)

# Evaluation
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Confusion matrix
preds = model.predict(test_generator)
predictions = np.where(preds >= 0.5, 1, 0)
cm = confusion_matrix(predictions, test_generator.classes)
print("Confusion Matrix:")
print(cm)

# Plotting
# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylim(0)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()