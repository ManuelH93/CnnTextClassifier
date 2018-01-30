# CnnTextClassifier
Character-level text classification using a convolutional neural network built with Keras.
 
Roughly inspired by Zhang et al. "Character-level Convolutional Networks for Text Classification" (https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). 

Implemented model uses a simplified architecture adapted for my use case (obfuscated text). Parameters and network topology probably need to be adjusted for your use case.

# Data
xtrain.txt - training examples  
ytrain.txt - training labels

xtest.txt - examples for which predictions are expected

# Requirements
python 3  
keras  
scikit-learn  
numpy  
