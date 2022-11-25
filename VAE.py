from keras import layers
from keras.models import Model
import keras

class VAE(Model):
  def __init__(self, layers_dims, img_shape):
    super(VAE, self).__init__()
    original_dim = img_shape[0] * img_shape[1]
    intermediate_dims = layers_dims[:-1]
    latent_dim = layers_dims[-1]

    inputs = keras.Input(shape=(original_dim,))
    h = inputs
    for intermediate_dim in intermediate_dims:
      h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_sigma = layers.Dense(latent_dim)(h)


    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + keras.backend.exp(z_log_sigma) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    self.encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs
    for intermediate_dim in reversed(intermediate_dims):
      x = layers.Dense(intermediate_dim, activation='relu')(x)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    self.decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = self.decoder(self.encoder(inputs)[2])
    self.vae = keras.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    self.vae.add_loss(vae_loss)
    self.vae.compile(optimizer='adam')

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded[2])
    return decoded