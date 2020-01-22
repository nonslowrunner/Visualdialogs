from dependencies import *

#Define an optimizer and Loss function

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
  
# Train 
@tf.function
def train_step(i_input, q_input, a_target, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(q_input, i_input, enc_hidden)

    dec_input = tf.expand_dims([tokenizer_a.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, a_target.shape[1]):
      # passing enc_output to the decoder
      predictions, enc_hidden, _ = decoder(dec_input, enc_hidden, enc_output)

      loss += loss_function(a_target[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(a_target[:, t], 1)

  batch_loss = (loss / int(a_target.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
  
  
  
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()
  
  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  
  for (batch, (i_input, q_input, a_target)) in enumerate(dataset_IQA.take(steps_per_epoch)):
    batch_loss = train_step(i_input, q_input, a_target, enc_hidden)
    total_loss += batch_loss
    

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
