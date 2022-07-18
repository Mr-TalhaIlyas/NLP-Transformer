import tensorflow as tf
from data_loader import tokenizers
from plotting import plot_attention_head, plot_attention_weights

MAX_TOKENS = 128

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # input sentence is portuguese, hence adding the start and end token
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is english, initialize the output with the
    # english start token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)
    text = tokenizers.en.detokenize(output)[0]  # shape: ()

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text, tokens, attention_weights

def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')

# transformer = tf.saved_model.load('./models/')

# transformer('este é o primeiro livro que eu fiz.').numpy()

# translator = Translator(tokenizers, transformer)

# sentence = 'este é um problema que temos que resolver.'
# ground_truth = 'this is a problem we have to solve .'

# translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
# print_translation(sentence, translated_text, ground_truth)

# head = 0
# # shape: (batch=1, num_heads, seq_len_q, seq_len_k)
# attention_heads = tf.squeeze(attention_weights['decoder_layer4_block2'], 0)
# attention = attention_heads[head]

# in_tokens = tf.convert_to_tensor([sentence])
# in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
# in_tokens = tokenizers.pt.lookup(in_tokens)[0]


# plot_attention_head(in_tokens, translated_tokens, attention)

# plot_attention_weights(sentence, translated_tokens,
#                        attention_weights['decoder_layer4_block2'][0])