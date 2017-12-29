from keras.datasets import mnist

(xtrain, _), (xtest,_)  = mnist.load_data()
# mnist verisini al

from keras.layers import Dense, Conv2DTranspose, Reshape, Conv2D,Flatten, BatchNormalization, Input,UpSampling2D,Dropout
from keras import Sequential,Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras import initializers
K.clear_session()

input_size = 100 # 100 random sayi
optimizer = Adam(lr=0.0002)
generator = Sequential()
generator.add(Dense(256,activation='relu',input_dim=input_size))
generator.add(Dense(512,activation='relu'))
generator.add(Dense(784,activation='tanh'))

discriminator = Sequential()
discriminator.add(Dense(512,activation='relu',input_dim=784))
discriminator.add(Dense(256,activation='relu'))
discriminator.add(Dense(1,activation='sigmoid'))
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

discriminator.trainable = False
x = Input((input_size,))
out_generator = generator(x)
out_discriminator = discriminator(out_generator)
gan = Model(inputs=(x,), outputs=(out_discriminator))
gan.compile(loss="binary_crossentropy", optimizer=optimizer)
gan.summary()

def generateRandomData(sizey, sizex):
    """ Can egrisi dagilimina sahip veri olustur """
    return np.random.normal(0,1,(sizey,sizex))

def showResults(gen):
    noise = generateRandomData(32,input_size)
    images = gen.predict(noise)
    plt.figure(figsize=(4,8))
    for i in range(32):
        plt.subplot(4,8,i+1)
        im = np.reshape(images[i],(1,-1))
        im = (np.reshape(im,[28,28])+1) *255
        im = np.clip(im,0,255)
        im = np.uint8(im)
        plt.imshow(im, cmap='gray')
    plt.show()   

epochs = 20
batch_size = 128
eval_size = 32
xtrain = np.reshape(xtrain,[xtrain.shape[0],-1])
xtest = np.reshape(xtest,[xtest.shape[0],-1])
xtrain = (xtrain.astype(np.float32) - 127.5)/127.5
plt.imshow(np.reshape(xtrain[0],[28,28] ),cmap="gray")
plt.show()

for e in range(epochs):
    for i in range(xtrain.shape[0]/batch_size):
        # gercek veriyi al
        xreal = xtrain[(i) * batch_size:(i+1)* batch_size]
        # fake veriyi olustur
        noise = generateRandomData(batch_size, input_size)
        # fake ciktiyi al
        xfake = generator.predict_on_batch(noise)
        # gercek ile egit
        discriminator.trainable = True
        discriminator.train_on_batch(xreal, np.array([0.9]*batch_size))
        # fake ile egit
        discriminator.train_on_batch(xfake, np.array([0.]*batch_size))
        # gan i egit
        discriminator.trainable = False
        gan.train_on_batch(noise, np.array([1.]*batch_size))
    
    # ekrana yazdir
    if (e+1) % 4 == 0:
        showResults(generator)