# Movie Sentiment Classification Project

A deep learning-based sentiment analysis project that classifies movie reviews as positive or negative using a Bidirectional LSTM neural network.

## 📋 Project Overview

This project implements a sentiment classification model for movie reviews using the IMDB dataset. The model uses a Bidirectional LSTM architecture with text vectorization to achieve high accuracy in distinguishing between positive and negative movie reviews.

## 🎯 Features

- **Dataset**: IMDB Reviews dataset (25,000 training + 25,000 test samples)
- **Model Architecture**: Bidirectional LSTM with embedding layer
- **Text Processing**: TensorFlow TextVectorization with 10,000 vocabulary tokens
- **Performance**: Achieves ~86% accuracy on test set
- **Visualization**: Training/validation accuracy and loss plots

## 🏗️ Model Architecture

The model consists of the following layers:

1. **TextVectorization Layer**: Converts text to numerical sequences (max 10,000 tokens)
2. **Embedding Layer**: 64-dimensional word embeddings
3. **Bidirectional LSTM Layer 1**: 64 units with return sequences
4. **Bidirectional LSTM Layer 2**: 32 units
5. **Dense Layer**: 64 units with ReLU activation
6. **Output Layer**: Single unit for binary classification

## 📊 Results

- **Training Accuracy**: 98.53%
- **Test Accuracy**: 86.07%
- **Training Loss**: 0.054
- **Test Loss**: 0.414

The model shows some overfitting (higher training accuracy than test accuracy), which is common in sentiment analysis tasks.


## 📈 Performance Analysis

The model training shows:
- Rapid improvement in the first 2 epochs
- Peak validation accuracy of ~87% at epoch 2
- Some overfitting after epoch 3
- Final test accuracy of 86.07%


## 📁 Project Structure

```
TextClassificationProject/
├── movie_sentiment_classification.ipynb    # Main notebook
├── README.md                               # This file
```

## 🛠️ Technical Details

- **Framework**: TensorFlow 2.x
- **Dataset**: IMDB Reviews (TensorFlow Datasets)
- **Batch Size**: 32
- **Epochs**: 5
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Vocabulary Size**: 10,000 tokens

## 🔧 Potential Improvements

1. **Reduce Overfitting**:
   - Add dropout layers
   - Implement early stopping
   - Use regularization techniques

2. **Model Enhancements**:
   - Try different architectures (GRU, Transformer)
   - Experiment with pre-trained embeddings (Word2Vec, GloVe)
   - Implement attention mechanisms

3. **Data Augmentation**:
   - Use data augmentation techniques
   - Implement cross-validation
   - Try different text preprocessing methods

## 📚 Dependencies

- `tensorflow` - Deep learning framework
- `tensorflow-datasets` - Dataset loading
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
