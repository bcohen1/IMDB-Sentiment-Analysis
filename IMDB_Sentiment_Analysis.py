import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import ShuffleSplit, cross_validate, train_test_split

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn import linear_model, svm, neighbors, gaussian_process, naive_bayes, tree, ensemble
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, \
    Embedding, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.utils import to_categorical

cwd = os.getcwd()
os.chdir(cwd)

train_df = pd.read_csv('imdb_train.csv')
test_df = pd.read_csv('imdb_test.csv')

combine = [train_df, test_df]

'''Analyze'''
train_df.info()
train_df.describe()

'''Pre-processing'''
# read in a small English language model
nlp = spacy.load("en_core_web_sm")

#Get list of stop words
stop_words = nlp.Defaults.stop_words

#Remove the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Remove special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def remove_stopwords(words):
    filtered_tokens = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    filtered_text = re.sub('\s{2,}', ' ', filtered_text)
    return filtered_text

for df in combine:
    df['review'] = np.vectorize(strip_html)(df['review'])
    df['review'] = np.vectorize(remove_special_characters)(df['review'])

    tokens = []
    lemma = []
    #Tokenize and lemmatize reviews
    for doc in nlp.pipe(df['review'], batch_size=50, disable=['ner','textcat']):
        if doc.has_annotation("DEP"):
            tokens.append([n.text for n in doc])
            lemma.append([n.lemma_ for n in doc])
        else:
            # We want to make sure that the lists of parsed results have the
            # same number of entries of the original Dataframe, so add some blanks in case the parse fails
            tokens.append(None)
            lemma.append(None)
    
    df['review_tokens'] = tokens
    df['review_lemma'] = lemma

    df['review_no_sw'] = np.vectorize(remove_stopwords)(df['review_lemma'])

norm_train_reviews=train_df['review_no_sw']
norm_test_reviews=test_df['review_no_sw']

#Label the sentiment (target) data
lb=LabelBinarizer()
sentiment_data=lb.fit_transform(train_df['sentiment'])

#Split train data for model evaluation purposes
X_train, X_test, y_train, y_test = train_test_split(norm_train_reviews, sentiment_data, test_size=0.25, random_state=42)

#Tfidf vectorizer
tv=TfidfVectorizer(ngram_range=(1,3))#min_df=0, max_df=1,
#Transform train & test reviews (feature)
tv_X_train=tv.fit_transform(X_train)
tv_X_test=tv.transform(X_test)

#Train the model
lr=LogisticRegression(max_iter=500, random_state=42)
#Fit the model for tfidf features
lr_tfidf=lr.fit(tv_X_train, y_train.ravel())
#Predict the model for tfidf features
lr_tfidf_predict=lr.predict(tv_X_test)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(y_test, lr_tfidf_predict)

#training the linear svm
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)
#fitting the svm for tfidf features
svm_tfidf=svm.fit(tv_X_train, y_train.ravel())
#Predicting the model for tfidf features
svm_tfidf_predict=svm.predict(tv_X_test)
#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(y_test, svm_tfidf_predict)

'''Actual test run'''
tv_X_true_test=tv.transform(norm_test_reviews)
Test_predict=lr.predict(tv_X_true_test)
submission_df = pd.concat([test_df['id'], pd.DataFrame(Test_predict)], axis=1)
submission_df.columns = ['id', 'sentiment']
submission_df['sentiment'] = np.where(submission_df['sentiment'] == 1, 'positive', 'negative')
submission_df.to_csv('Test_Submission.csv', index=False)

#Classification report for tfidf features
lr_tfidf_report=classification_report(y_test, lr_tfidf_predict, target_names=['Positive','Negative'])
print(lr_tfidf_report)

#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,lr_tfidf_predict,labels=[1,0])
print(cm_tfidf)

'''Create word embedding matrix'''
# get a count of the number of possible categories to predict
num_classes = len(np.unique(y_train))

# convert the training and testing dataset
# each label is one-hot encoded into a vector of size num_classes
y_train_array = to_categorical(y_train, num_classes)
y_test_array = to_categorical(y_test, num_classes)

# prepare tokenizer
MAX_NUM_WORDS = 25000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

# learn the vocabulary from the training documents
tokenizer.fit_on_texts(X_train)

# increment the vocab count by one as the first embedding must be blank for later activities in Keras
vocab_size = len(tokenizer.word_index) + 1

#convert the sentences to a sequence of token ids
encoded_docs = tokenizer.texts_to_sequences(X_train)

# pad documents to length of max value
max_seq_length = len(max(encoded_docs, key=len))

# encode both train and test data (only learn from the training data)
# Convert array of text into a series of padded (with 0's) token interger-ids 
def encode_text(text):
    encoded_docs = tokenizer.texts_to_sequences(text)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_seq_length, padding='post')    
    return padded_docs

X_train_sequence = encode_text(X_train)
X_test_sequence = encode_text(X_test)

#prepare an "embedding matrix" which will contain at index i the embedding vector 
#for the word of index i in our word index.

#Download pre-trained Glove word embedding
# !wget http://nlp.stanford.edu/data/glove.840B.300d.zip

# Glove Word Embeddings
GLOVE_DIR = cwd + '\\glove.840B.300d\\glove.840B.300d.txt'
embedding_size = 300

# Store all embeddings {'token': n-dimensional embedding_series}
embeddings_index = {}

with open(GLOVE_DIR, 'rb') as f:
    for line in f:
        values = line.split()
        word = values[0].decode('utf-8')
        embedding = np.asarray(values[1:], dtype='float32')

        # store the embeddings in a dict
        embeddings_index[word] = embedding

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_size))

for word, i in tokenizer.word_index.items():    
    embedding_vector = embeddings_index.get(word)
    
    # add each word in the embedding_matrix in the slot for the tokenizer's word id
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
def create_embeddings_baseline_for_lr(data_sequences):
    """ create a baseline by averaging all the embeddings in a sentence. This provides a single
    array to describe the document, which allows the use of standard statistical models (e.g. logistic regression)
    
    :param data_sequences: 2d-list, list of lists contains id mappings for words in a sentence 
    """
    
    # store all averaged embeddings for all sentences
    averaged_sentence_embeddings = []
    
    # iterate through all sentences
    for sentence_sequence in data_sequences:
        
        # convert id sequence into embeddings for one sentence
        embeddings = [] # store embedding of one sentence
        for word_id in sentence_sequence:
            embedding = embedding_matrix[word_id]
            embeddings.append(embedding)

        # average the embeddings of one sentence
        avg_embeddings = pd.DataFrame(embeddings).mean().values
        averaged_sentence_embeddings.append(avg_embeddings)

    # store all averaged embeddings in a dataframe
    avg_emb_df = pd.DataFrame(averaged_sentence_embeddings)
    
    return avg_emb_df

X_train_avg_emb_df = create_embeddings_baseline_for_lr(X_train_sequence)
X_test_avg_emb_df = create_embeddings_baseline_for_lr(X_test_sequence)

# fit model for word embeddings
logreg = LogisticRegression(solver='lbfgs', max_iter=500, multi_class='auto')
logreg.fit(X_train_avg_emb_df, y_train.ravel())

# store predictions
y_pred = logreg.predict(X_test_avg_emb_df)

# get model score
lr_score=accuracy_score(y_test, y_pred)

'''Build NN'''
embedding = Embedding(
    input_dim=vocab_size, 
    output_dim=embedding_size,                                    
    input_length=max_seq_length,
    embeddings_initializer=Constant(embedding_matrix),
    trainable=False                                   
)

# build a convolutional neural network (CNN) ending in a softmax output
# keras sequential creates models layer-by-layer, doesn't create models that share layers or have multiple inputs/outputs
lstm_model = Sequential()

# load the pretrained embedding into the model
lstm_model.add(embedding)

# create a 1D (Conv1D) convolutional layer for text (2D is for images)
# filters: the number of features to extract from the text
# kernel_size: the window size (how many words to look at per feature)
lstm_model.add(Conv1D(filters=1024, kernel_size=6, activation='relu'))    

# final pooling before dense layer
lstm_model.add(GlobalMaxPooling1D())

# dense layers for a feedforward neural network
lstm_model.add(Dense(num_classes, activation='softmax'))

# compile model to set the optimizer, loss, and metrics
lstm_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# set-up and train model
# cnn_model = build_cnn_model(embedding_model)

# train model
lstm_model.fit(
    X_train_sequence, 
    y_train_array,
    epochs=3,
    shuffle=True,
    validation_data=(X_test_sequence, y_test_array)
)

'''Actual test run'''
X_true_test = encode_text(norm_test_reviews)
Test_predict = lstm_model.predict_classes(X_true_test)
submission_df = pd.concat([test_df['id'], pd.DataFrame(Test_predict)], axis=1)
submission_df.columns = ['id', 'sentiment']
submission_df['sentiment'] = np.where(submission_df['sentiment'] == 1, 'positive', 'negative')
submission_df.to_csv('Test_Submission.csv', index=False)

lstm_model.save('lstm_model.h5')
