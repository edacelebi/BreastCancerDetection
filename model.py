from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, MaxPool2D
from keras.layers import Dense, Dropout, AveragePooling2D
from keras.layers import Flatten
from keras.optimizers import Adam, SGD

class model:

    def lenet5(self):
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                         input_shape=(32, 32, 1), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(120, activation='tanh'))
        model.add(Dense(84, activation='tanh'))
        model.add(Dense(3, activation='softmax'))

        opt = SGD(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def alexNet(self):
        model =Sequential([
            Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(128, 128, 1)),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])

        opt = SGD(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def VGG16(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 1), padding='same',
                   activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same' ))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same' ))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same' ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same' ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same' ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same' ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same' ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same' ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same' ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model




