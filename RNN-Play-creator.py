from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#Para usar tu propio guion o texto, descomenatos el siguiente codigo:

# from google.colab import files
# path_to_file = list(files.upload().keys())[]


text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(len(text)) #La cantidad de palabras en el texto
print(text[:300])

#Agregamos un numero a cada caracter

vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

print('Text: ', text[:15])
print('Codificado: ', text_to_int(text[:15]))

def int_to_text(int):
    try:
        int = int.numpy()
    except:
        pass
    return ''.join(idx2char[int])

#Hacemos pequennos trainings, ya que no es adecuado pasar el millon de letras a nuestra trainings

seq_length = 100
examples_per_epoch = len(text)//(seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#Creamos el modelo, para este usaremos LSTM con una dense layer.

def build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                                    batch_input_shape = [BATCH_SIZE, None]),
        tf.keras.layers.LSTM(RNN_UNITS,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(VOCAB_SIZE)
    ])
    return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

#Tenemos que crear nuesta propia loss function.

for input_example_batch, targer_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (batch_size, sequence_lenght, vocab_size)')

#comenta y descomenta las siguientes lienas para tener una idea.
# print(len(example_batch_predictions))
# print(example_batch_predictions)
#
# pred = example_batch_predictions[0]
# print(len(pred))
# print(pred)
#
# time_pred =  pred[0]
# print(len(time_pred))
# print(time_pred)
#
# sampled_indices = tf.random.categorical(pred, num_samples = 1)
#
# sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
# predicted_chars = int_to_text(sampled_indices)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoints_dir = '../training_checkpoint'
checkpoint_prefix = os.path.join(checkpoints_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only=True)

history = model.fit(data, epochs=40, callbacks = [checkpoint_callback])

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size = 1)

model.load_weights(tf.train.lastest_checkpoint(checkpoints_dir))
model.build(tf.TensorShape([1, None]))

def text_generator(model, start_string):
    num_generate = 800

    input_eval = [char2idx[s] for s in start_string]
    input-eval = tf.expand_dims(input_eval, 0)

    text_generated =[]
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)
        predictions = predictions/temperature

        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))
