  # Combined Generator loss (segmentation loss on source and enchanced source batches + conditional generator loss):
  def __gen_loss(self, pred, prediction_gen, fake_output, labels, step):
      k = 2 if step < 1000 else 1
      pred_loss = self.classifier_loss_function(labels, pred)
      pred_gen_loss = self.classifier_loss_function(labels, prediction_gen)
      gen_loss = self.bce_loss_function(tf.ones_like(fake_output), fake_output)
      total_gen_loss = pred_loss + pred_gen_loss + k*gen_loss
      return total_gen_loss

  # Discriminator loss:
  def __disc_loss(self, source_out, target_out):
      target_loss = self.bce_loss_function(tf.ones_like(target_out), target_out)
      source_enc_loss = self.bce_loss_function(tf.zeros_like(source_out), source_out)
      total_loss = 0.5 * (target_loss + source_enc_loss)
      return total_loss
  
  # Train step:
  @tf.function
  def __train_step(self, source_batch, labels_batch, target_batch, epoch, step):

      weights = [var for var in self.VGG_encoder_decoder.trainable_variables
                 if "decoder" in var.name or "classifier" in var.name]

      # First step:
      with tf.GradientTape(watch_accessed_variables=False) as gen_tape,\
           tf.GradientTape(watch_accessed_variables=True) as disc_tape:
          gen_tape.watch(weights)
          
          prediction, source_encoded = self.VGG_encoder_decoder(source_batch,  training=True, use_generator=False)
          prediction_gen, fake_features = self.VGG_encoder_decoder(source_batch, training=True, use_generator=True)
          _, true_features = self.VGG_encoder_decoder(target_batch, training=True, use_generator=False)

          real_output = tf.math.sigmoid(self.discriminator(true_features, training=True))
          fake_output = tf.math.sigmoid(self.discriminator(fake_features, training=True))

          loss_gen = self.__gen_loss(prediction, prediction_gen, fake_output, labels_batch, (epoch+1)*step)
          loss_disc = self.__disc_loss(fake_output, real_output)

      g_grads = gen_tape.gradient(loss_gen, weights)
      self.gen_optimizer.apply_gradients(zip(g_grads, weights))

      d_grads = disc_tape.gradient(loss_disc, self.discriminator.trainable_weights)
      self.disc_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

      self.train_disc_loss.update_state(loss_disc)
      self.train_gen_loss.update_state(loss_gen)

      # Second step:
      weights = [var for var in self.VGG_encoder_decoder.trainable_variables
                 if "encoder_module" in var.name or "generator_module" in var.name]

      with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
          gen_tape.watch(weights)

          prediction, _ = self.VGG_encoder_decoder(source_batch,  training=True, use_generator=False)
          prediction_gen, fake_features = self.VGG_encoder_decoder(source_batch, training=True, use_generator=True)
          _, true_features = self.VGG_encoder_decoder(target_batch, training=True, use_generator=False)

          fake_output = tf.math.sigmoid(self.discriminator(fake_features, training=True))

          loss_gen = self.__gen_loss(prediction, prediction_gen, fake_output, labels_batch, (epoch+1)*step)

      grads = gen_tape.gradient(loss_gen, weights)
      self.gen_optimizer.apply_gradients(zip(grads, weights))
