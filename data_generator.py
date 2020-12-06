from keras.preprocessing.image import ImageDataGenerator

data = "asl_alphabet/asl_alphabet_train/asl_alphabet_train/"
validation_split = 0.1
target_size = (32, 32)
batch_size = 64
color_mode = 'rgb'

data_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=validation_split,
)

train_data_generate = data_generator.flow_from_directory(
    data,
    target_size=target_size,
    batch_size=batch_size,
    color_mode=color_mode,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

validation_data_generate = data_generator.flow_from_directory(
    data,
    target_size=target_size,
    batch_size=batch_size,
    color_mode=color_mode,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

# print(train_data_generate.class_indices)
