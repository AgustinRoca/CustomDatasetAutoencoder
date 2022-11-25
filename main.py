import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os
from VAE import VAE

def download_dataset(
  dataset_url: str,
  img_shape = (28,28),
  validation_split: float = 0.2,
  seed: int = 123):

  data_dir = str(pathlib.Path.home()) + '/.keras/datasets/' + dataset_url.split('.')[-2].split('/')[-1]

  if not os.path.isdir(data_dir):
    tf.keras.utils.get_file(origin=dataset_url, extract=True)

  data_path = pathlib.Path(data_dir)

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    image_size=img_shape,
    labels=None,
    validation_split=validation_split,
    subset="training",
    seed=seed,
    color_mode='grayscale')

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    image_size=img_shape,
    labels=None,
    validation_split=validation_split,
    subset="validation",
    seed=seed,
    color_mode='grayscale')

  train_ds = dataset_to_numpy(train_ds)
  validation_ds = dataset_to_numpy(validation_ds)

  return train_ds, validation_ds

def preprocess_dataset(dataset):
  dataset = dataset.astype('float32') / 255.
  dataset = dataset.reshape((len(dataset), np.prod(dataset.shape[1:])))
  return dataset  

def dataset_to_numpy(dataset):
  data = np.array(list(dataset.as_numpy_iterator()))
  samples = []
  for i in range(len(data)):
    samples += list(data[i])
  samples = np.array(samples)[:,:,:,0]
  return samples

def show_images(original, reconstructed, img_shape, n: int = 10):
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # Display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(original[i].reshape(img_shape))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(reconstructed[i].reshape(img_shape))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()

def show_latent_space(models, n: int = 15, ):
  figure = np.zeros((img_shape[0] * n, img_shape[1] * n))
  # We will sample n points within [-15, 15] standard deviations
  grid_x = np.linspace(-3, 3, n)
  grid_y = np.linspace(-3, 3, n)

  for i, yi in enumerate(grid_x):
      for j, xi in enumerate(grid_y):
          z_sample = np.array([[xi, yi]])
          x_decoded = models.decoder.predict(z_sample)
          digit = x_decoded[0].reshape(img_shape)
          figure[i * img_shape[0]: (i + 1) * img_shape[1],
                j * img_shape[0]: (j + 1) * img_shape[1]] = digit

  plt.figure(figsize=(10, 10))
  plt.imshow(figure)
  plt.show()


# dataset_url = "https://github.com/DeepLenin/fashion-mnist_png/raw/master/data.zip"
# dataset_url = "https://archive.org/download/ffhq-dataset/thumbnails128x128.zip"
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_url = "https://s3.amazonaws.com/nist-srd/SD19/by_merge.zip"
img_shape = (28, 28)

x_train, x_test = download_dataset(dataset_url, img_shape=img_shape)
x_train = preprocess_dataset(x_train)
x_test = preprocess_dataset(x_test)

models = VAE([64, 2], img_shape)

models.vae.fit(x_train, x_train,
                epochs=100,
                batch_size=32,
                validation_data=(x_test, x_test))

encoded_imgs = models.encoder.predict(x_test)
decoded_imgs = models.decoder.predict(encoded_imgs[2])
show_images(x_test, decoded_imgs, img_shape=img_shape)
show_latent_space(models)