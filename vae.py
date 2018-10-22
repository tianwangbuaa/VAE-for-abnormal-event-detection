x = Input(shape=(original_dim,))
h = Dense(output_dim=intermediate_dm, activation='relu', name='Dense1')(x)
z_mean = Dense(output_dim=latent_dim, name='Dense2')(h)
z_log_var = Dense(output_dim=latent_dim, name='Dense3')(h)


def sampling(args):
   z_mean, z_log_var = args
   epsilon = K.random_normal(shape=(batch_size_set, latent_dim), mean=0, std=epsilon_std)

   return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])

decoder_h = Dense(intermediate_dm, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
   xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
   kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
   return xent_loss + kl_loss

model = Model(x, x_decoded_mean)
