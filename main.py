# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:57:32 2022

@author: talha
"""

import tensorflow as tf

# Import tf_text to load the ops used by the tokenizer saved model
import tensorflow_text  # pylint: disable=unused-import

from data_loader import tokenizers, train_batches, val_batches, train_samples
from transformer import Transformer
from callbacks import CustomSchedule
#from infer import Translator#, translator
from loss_metrics  import loss_function, accuracy_function
from tqdm import trange

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 20
MAX_TOKENS = 128
BATCH_SIZE = 64
checkpoint_path = './chkpt/'


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    rate=dropout_rate)


ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
#%%
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1] # TF input /END token ??
    tar_real = tar[:, 1:] # for loss calculation/START token removed
    
    with tf.GradientTape() as tape:
      predictions, _ = transformer([inp, tar_inp],
                                   training = True)
      loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))
#%%
acc, loss = [], []
training_steps = int(EPOCHS*(train_samples/BATCH_SIZE))
t = trange(training_steps, desc='Training Transformer (Filling Data Buffers)', leave=True)
for epoch in t:

    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)
      
        t.set_description(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}\Accuracy {train_accuracy.result():.4f}')
        t.update()
        
    acc.append(train_accuracy.result())
    loss.append(train_loss.result())
        
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()

#%%


inp = [ 'o que você disse!']
inpt= tf.convert_to_tensor(inp, dtype=tf.string)

sentence = tokenizers.pt.tokenize(inpt).to_tensor()

encoder_input = sentence

start_end = tokenizers.en.tokenize([''])[0]
start = start_end[0][tf.newaxis]
end = start_end[1][tf.newaxis]

output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True,
                              clear_after_read=False)
output_array = output_array.write(0, start)


for i in tf.range(128):
      output = tf.transpose(output_array.stack())
      predictions, _ = transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])
      tt = tf.transpose(output_array.stack())
      text = tokenizers.en.detokenize(tt)[0]
      print(text.numpy())
      if predicted_id == end:
        break
#%%
from infer import Translator, print_translation

translator = Translator(tokenizers, transformer)

text, tokens, attention_weights = translator(tf.constant(inpt))

print_translation(text, tokens, 'None')
#%%
'''
Export/Saving Model
'''

class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence, max_length=MAX_TOKENS):
    (result,
     tokens,
     attention_weights) = self.translator(sentence)

    return result

translator = ExportTranslator(translator)

translator(tf.constant('este é o primeiro livro que eu fiz.')).numpy()

tf.saved_model.save(translator, export_dir='./models/')