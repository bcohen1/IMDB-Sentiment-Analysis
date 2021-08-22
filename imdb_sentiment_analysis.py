import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import regex as re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dropout,
    Dense
)


def strip_html(text):
    """Remove html tags"""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_special_characters(text, remove_digits=True):
    """Remove special characters"""
    pattern = r"[^a-zA-z0-9\s]"
    text = re.sub(pattern, "", text)
    return text


def lemmatize(words):
    """Get base words (lemmatize) from tokenized words"""
    lemma = []
    for doc in nlp.pipe(words, batch_size=50, disable=["ner", "textcat"]):
        if doc.has_annotation("DEP"):
            lemma.append([n.lemma_ for n in doc])
        else:
            # Add blanks in case the parse fails to ensure the lists of parsed results have the same number of entries as the original Dataframe
            lemma.append(None)
    return lemma


def remove_stopwords(words):
    """Remove stop words according to Spacy"""
    filtered_tokens = [word for word in words if word not in stop_words]
    filtered_text = " ".join(filtered_tokens)
    filtered_text = re.sub(r"\s{2,}", " ", filtered_text)
    return filtered_text


def encode_text(text, max_seq_length):
    """Convert array of text into a series of padded (with 0's) token interger-ids"""
    encoded_docs = tokenizer.texts_to_sequences(text)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_seq_length, padding="post")
    return padded_docs


def extract_word_embedding(vocab_size, EMBEDDING_SIZE):
    """Create word embedding matrix"""
    glove_dir = r"glove.840B.300d.txt"

    # Store all embeddings {'token': n-dimensional embedding_series}
    embeddings_index = {}

    with open(glove_dir, "rb") as f:
        for line in f:
            values = line.split()
            word = values[0].decode("utf-8")
            embedding = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = embedding

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)

        # add each word in the embedding_matrix in the slot for the tokenizer's word id
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_model(vocab_size, EMBEDDING_SIZE, max_seq_length, embedding_matrix):
    """Build neural network using Glove pretrained word embedding"""
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_SIZE,
        input_length=max_seq_length,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )

    model = Sequential()

    # Load the pretrained embedding into the model
    model.add(embedding)

    # Create a 1D (Conv1D) convolutional layer
    # Filters: the number of features to extract from the text
    # Kernel_size: the window size (how many words to look at per feature)
    model.add(Conv1D(filters=100, kernel_size=3, activation="relu"))

    # Dropout layer to prevent overfitting
    model.add(Dropout(0.25))

    # Final pooling before dense layer
    model.add(GlobalMaxPooling1D())

    # Dense layers for a feedforward neural network
    model.add(Dense(num_classes, activation="softmax"))

    # Compile model to set the optimizer, loss, and metrics
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


df = pd.read_csv("IMDB Dataset.csv", encoding="utf-8")

# read in a small English language model
nlp = spacy.load("en_core_web_sm")

# Get list of stop words
stop_words = nlp.Defaults.stop_words

df["review"] = np.vectorize(strip_html)(df["review"])
df["review"] = np.vectorize(remove_special_characters)(df["review"])

df["review_lemma"] = lemmatize(df["review"])
df["review_lemma"] = np.vectorize(remove_stopwords)(df["review_lemma"])

# Label the sentiment (target) data
lb = LabelBinarizer()
df["sentiment"] = lb.fit_transform(df["sentiment"])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    df["review_lemma"], df["sentiment"], test_size=0.2, random_state=42
)

# Split train data for model evaluation purposes
x_train, x_cv, y_train, y_cv = train_test_split(
    x_train, y_train, test_size=0.25, random_state=42
)

# Convert tokens to term frequency-inverse document frequency (TFIDF)
tv = TfidfVectorizer(ngram_range=(1, 3))
tv_x_train = tv.fit_transform(x_train)
tv_x_cv = tv.transform(x_cv)

# Use Logisitc Regression model as baseline for CNN model
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(tv_x_train, y_train.ravel())
lr_tfidf_predict = lr.predict(tv_x_cv)

lr_tfidf_report = classification_report(
    y_cv, lr_tfidf_predict, target_names=["Positive", "Negative"]
)
print(lr_tfidf_report)

num_classes = len(np.unique(y_train))

# Each label is one-hot encoded into a vector of size num_classes
y_train_encoded = to_categorical(y_train, num_classes)
y_cv_encoded = to_categorical(y_cv, num_classes)

tokenizer = Tokenizer(num_words=25000)

# learn the vocabulary from the training documents
tokenizer.fit_on_texts(x_train)

# convert the sentences to a sequence of token ids
encoded_docs = tokenizer.texts_to_sequences(x_train)

# pad documents to length of max value
max_seq_length = len(max(encoded_docs, key=len))

x_train_encoded = encode_text(x_train, max_seq_length)
x_cv_encoded = encode_text(x_cv, max_seq_length)

# increment the vocab count by one as the first embedding must be blank
vocab_size = len(tokenizer.word_index) + 1
EMBEDDING_SIZE = 300

embedding_matrix = extract_word_embedding(vocab_size, EMBEDDING_SIZE)

cnn_model = build_model(vocab_size, EMBEDDING_SIZE, max_seq_length, embedding_matrix)

try:
    reconstructed_model = load_model("cnn_model")
except:
    cnn_model.fit(
        x_train_encoded,
        y_train_encoded,
        batch_size=50,
        epochs=3,
        shuffle=True,
        validation_data=(x_cv_encoded, y_cv_encoded),
    )

    # Actual test run
    x_test_encoded = encode_text(x_test, max_seq_length)
    test_predict = np.argmax(cnn_model.predict(x_test_encoded), axis=-1)

    cnn_report = classification_report(
        y_test, test_predict, target_names=["Positive", "Negative"]
    )
    print(cnn_report)

    cnn_model.save("cnn_model")
