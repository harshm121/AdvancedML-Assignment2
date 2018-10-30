from keras import models
from keras import layers
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
#Load the VGG model
inet = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224, 3))

for layer in resnet.layers[:-4]:
    layer.trainable = False

resnet.summary();
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(inet)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()
batch_size = 16;
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
 
train_generator = train_datagen.flow_from_directory(
        './train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        './validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Train the model

model.fit_generator(
        train_generator,
        steps_per_epoch=1400 // batch_size,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=480 // batch_size)
 
# Save the model
model.save('inet.h5')