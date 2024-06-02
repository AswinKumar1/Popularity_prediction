# Song Popularity Prediction and Recommendation System

## Project Overview

This project aims to predict the popularity of songs based on their features and provide recommendations accordingly. The system leverages various machine learning techniques, including clustering and logistic regression, to classify genres and recommend songs. The end goal is to build a robust model that can predict song popularity and recommend songs similar to user-defined seed songs.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Models and Methods](#models-and-methods)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/song-popularity-prediction.git
   cd song-popularity-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset used in this project contains various features of songs, including:
- Danceability
- Duration (in milliseconds)
- Tempo
- Popularity
- Loudness
- Instrumentalness

Ensure that the dataset is available at the specified path or update the path in the code accordingly.

## Features

The features used for clustering and logistic regression are:
- `danceability`
- `duration_ms`
- `tempo`
- `popularity`
- `loudness`
- `instrumentalness`

These features are selected based on their relevance to the song's characteristics and potential impact on its popularity.

## Models and Methods

### 1. K-Means Clustering
- **Purpose:** To classify songs into different genres based on their features.
- **Implementation:** 
  ```python
  kmeans = KMeans(n_clusters=10)
  genre_labels = kmeans.fit_predict(df[number_cols])
  ```

### 2. Logistic Regression
- **Purpose:** To predict the popularity of songs.
- **Implementation:** 
  ```python
  log_reg = LogisticRegression(max_iter=10000)
  log_reg.fit(X_train_upsampled, y_train_upsampled)
  ```

### 3. Upsampling
- **Purpose:** To address class imbalance in the dataset.
- **Implementation:** 
  ```python
  X_train_upsampled, y_train_upsampled = resample(X_train, y_train, replace=True, n_samples=X_train.shape[0], random_state=42)
  ```

### 4. Evaluation Metrics
- **Classification Report**
- **ROC Curve and AUC**

## Evaluation

The model's performance is evaluated using the following metrics:
- **Classification Report:** Provides precision, recall, and F1-score for each class.
- **ROC Curve and AUC:** Plots the ROC curve for each class and computes the area under the curve.

Example of generating classification report and ROC curve:
```python
y_pred = log_reg.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

y_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

## Usage

To use the recommendation system, define a list of seed songs and call the recommendation function:
```python
seed_songs = [
    {'name': 'Come As You Are', 'year': 1991},
    {'name': 'Smells Like Teen Spirit', 'year': 1991},
    {'name': 'Lithium', 'year': 1992},
    {'name': 'All Apologies', 'year': 1993},
    {'name': 'Stay Away', 'year': 1993}
]

recommended_songs = recommend_songs_with_log_reg(seed_songs, df)
for idx, song in enumerate(recommended_songs.iterrows(), start=1):
    print(f"{idx}. {song[1]['name']} by {song[1]['artists']} ({song[1]['year']})")
```

## Results

The system provides a list of recommended songs based on the logistic regression model's predictions. The recommendations are ranked by predicted popularity probabilities.

## Conclusion

This project demonstrates the application of machine learning techniques in predicting song popularity and providing recommendations. By combining clustering and logistic regression, the system can effectively classify genres and predict song popularity, offering valuable insights and recommendations to users.

---

This README highlights your technical knowledge and the methods used in the project, providing a clear and comprehensive overview of the system.
