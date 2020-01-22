def evaluate(image_path, sentence):
  
  image_input =  load_image_vgg(image_path)
  
  image_input = tf.expand_dims(image_input[0], 0)
  
  inputs_I = image_features_extract_model_vgg(image_input)
  
  inputs_Q = tf.convert_to_tensor(sentence)
  
  inputs_Q = tf.expand_dims(inputs_Q, 0)
  
  result = ''
  
  hidden = [tf.zeros((1, units))]
  
  enc_out, enc_hidden = encoder(inputs_Q, inputs_I , hidden)
  
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([tokenizer_a.word_index['<start>']], 0)
  
  for t in range(max_length_a):
    predictions , dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    
    # storing the attention weigths to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    
    predicted_id = tf.argmax(predictions[0]).numpy()
    
    result += tokenizer_a.index_word[predicted_id] + ' '
    
    if tokenizer_a.index_word[predicted_id] == '<end>':
      return result, sentence
                              
    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence                                
