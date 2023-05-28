from attention_keras.layers.attention import AttentionLayer


attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

inputs = Input(shape=(input_dims,))
attention_probs = Dense(input_dims, activation='softmax',
name='attention_probs')(inputs)
attention_mul = merge([inputs, attention_probs],
output_shape=32, name='attention_mul', mode='mul'