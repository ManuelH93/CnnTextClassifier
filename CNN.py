import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Activation, Dense, Dropout, Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.cross_validation import train_test_split

#---------------
# preprocessing
#---------------
# Load data
lines_train = [line.rstrip('\n') for line in open('xtrain_obfuscated.txt')]
lines_test = [line.rstrip('\n') for line in open('xtest_obfuscated.txt')]
labels_train = [line.rstrip('\n') for line in open('ytrain.txt')]

# Get Unique characters and labels
char_list = list(set(''.join(lines_train + lines_test)))
labels_list = list(set(labels_train))

# Remove possible duplicates
unique_train = []
for example in list(zip(lines_train, labels_train)):
    if len(example[0].strip()) != 0 and example not in unique_train:
        unique_train.append(example)

# Get indices for chars
char_indices = dict((c, i) for i, c in enumerate(char_list))
indices_char = dict((i, c) for i, c in enumerate(char_list))

# Get max length of sequences
MAX_LENGTH = 0
for l in (lines_train + lines_test):
    if len(l) > MAX_LENGTH:
        MAX_LENGTH = len(l)

# Function to convert char sequences to int sequences
def line_to_char_seq(line):
    line_chars = list(line)
    line_chars_indices = list(map(lambda char: char_indices[char], line_chars))
    return sequence.pad_sequences([line_chars_indices], maxlen=MAX_LENGTH)[0]

X = []
y = []
X_predict = []

# Convert sequences
for li, la in zip(lines_train, labels_train):
    X.append(line_to_char_seq(li))
    y.append(la)

for li in lines_test:
    X_predict.append(line_to_char_seq(li))

X = np.array(X).astype(np.uint8)
y = np_utils.to_categorical(np.array(y)).astype(np.bool)
X_predict = np.array(X_predict).astype(np.uint8)

print(X.shape, y.shape)

X_train = X
y_train = y

# Train test split used for validation during parameter tuning
# --> not used in final model to include as much data as possible
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#---------------------------------
# Model construction and training
#---------------------------------

batch_size = 256
epochs = 100

model = Sequential()
# input/char embedding layer
model.add(Embedding(len(char_list), 32, input_length=MAX_LENGTH, mask_zero=False))
# conv1D + maxpooling layer
model.add(Conv1D(1024, 7, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.25))
model.add(Conv1D(1024, 7, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.25))
# fully-connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
# softmax output layer
model.add(Dense(len(labels_list)))
model.add(Activation('softmax'))


rmsp = optimizers.RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=rmsp, metrics=['accuracy'])

early_stopping = EarlyStopping(patience=3, verbose=1)
checkpointer = ModelCheckpoint(filepath='sap_offline_challenge.hdf5', 
                               verbose=1, 
                               save_best_only=True)

model.fit(X_train, y_train, 
          batch_size=batch_size, 
          epochs = epochs, 
          verbose=1,
          shuffle=True,
          validation_split=0.1,
          callbacks=[early_stopping, checkpointer])


#-------------
# Predictions
#-------------

#model.load_weights('sap_offline_challenge.hdf5')

y_predict = model.predict_classes(X_predict)

# Save predictions
np.savetxt('ytest.txt', y_predict, fmt='%i' , newline="\n")

print(y_predict)
