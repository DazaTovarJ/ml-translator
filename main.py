import logging
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from layer.common.embedding import PositionalEmbedding
from model.loss_management import masked_accuracy, masked_loss
from model.optimizer import CustomSchedule

from model.transformer import Transformer
from translator import ExportTranslator, Translator, print_translation

logging.getLogger("tensorflow").setLevel(logging.ERROR)

examples, metadata = tfds.load(
    "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)
train_examples, val_examples = examples["train"], examples["validation"]


# for pt_examples, en_examples in train_examples.batch(3).take(1):
#     print("> Examples in Portuguese")
#     for pt in pt_examples.numpy():
#         print(pt.decode("utf-8"))
#     print()

#     print("> Examples in English")
#     for en in en_examples.numpy():
#         print(en.decode("utf-8"))

model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir=".",
    cache_subdir="",
    extract=True,
)

tokenizers = tf.saved_model.load(model_name)

# print([item for item in dir(tokenizers.en) if not item.startswith("_")])

# print("> This is a batch of strings:")
# for en in en_examples.numpy():
#     print(en.decode("utf-8"))

# encoded = tokenizers.en.tokenize(en_examples)

# print("> This is a padded-batch of token IDs:")
# for row in encoded.to_list():
#     print(row)

# round_trip = tokenizers.en.detokenize(encoded)

# print("> This is human-readable text:")
# for line in round_trip.numpy():
#     print(line.decode("utf-8"))

# print("> This is the text split into tokens:")
# tokens = tokenizers.en.lookup(encoded)
# print(tokens)

# lengths = []

# for pt_examples, en_examples in train_examples.batch(1024):
#     pt_tokens = tokenizers.pt.tokenize(pt_examples)
#     lengths.append(pt_tokens.row_lengths())

#     en_tokens = tokenizers.en.tokenize(en_examples)
#     lengths.append(en_tokens.row_lengths())
#     print(".", end="", flush=True)

# all_lengths = np.concatenate(lengths)

# plt.hist(all_lengths, np.linspace(0, 500, 101))
# plt.ylim(plt.ylim())
# max_length = max(all_lengths)
# plt.plot([max_length, max_length], plt.ylim())
# plt.title(f"Maximum tokens per example: {max_length}")
# plt.show()

MAX_TOKENS = 128


def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, : (MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 32


def make_batches(ds):
    return (
        ds.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (pt, en), en_labels in train_batches.take(1):
    break

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate,
)

output = transformer((pt, en))

print(en.shape)
print(pt.shape)
print(output.shape)

attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)

print(transformer.summary())

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
transformer.fit(train_batches, epochs=20, validation_data=val_batches)

sentence = "este é um problema que temos que resolver."
ground_truth = "this is a problem we have to solve ."

translator = Translator(tokenizers, transformer)

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence)
)
print_translation(sentence, translated_text, ground_truth)

translator = ExportTranslator(translator)

tf.saved_model.save(translator, export_dir="translator")

reloaded = tf.saved_model.load("translator")

reloaded('este é o primeiro livro que eu fiz.').numpy()
