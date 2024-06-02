import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Load the data
df = pd.read_csv('/content/data.csv')

# Define the columns to be used for clustering and logistic regression
number_cols = ['danceability', 'duration_ms', 'tempo', 'popularity','loudness','instrumentalness']  # Add or remove attributes as needed

# K-Means Clustering for Genre Classification
kmeans = KMeans(n_clusters=10)  # Specify the number of clusters
genre_labels = kmeans.fit_predict(df[number_cols])

# Logistic Regression for Recommendation based on Song Popularity
X = df[number_cols[:-1]]  # Features (excluding 'popularity')
y = df['popularity']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[number_cols], genre_labels, test_size=0.2, random_state=42)

# Upsampling the minority classes to address class imbalance
X_train_upsampled, y_train_upsampled = resample(X_train, y_train, replace=True, n_samples=X_train.shape[0], random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_upsampled, y_train_upsampled)

# Classification Report
y_pred = log_reg.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability estimates of the positive class
num_classes = len(np.unique(y_test))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
   fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob, pos_label=i)
   roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area if there are only 2 unique classes
if len(np.unique(y_test)) == 2:
   fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())
   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


seed_songs = [{'name': 'Come As You Are', 'year':1991},
               {'name': 'Smells Like Teen Spirit', 'year': 1991},
               {'name': 'Lithium', 'year': 1992},
               {'name': 'All Apologies', 'year': 1993},
               {'name': 'Stay Away', 'year': 1993}
             ]

def get_song_data(song_name, data):
   return data[data['name'] == song_name][number_cols].iloc[0]

# Recommendation Function
def recommend_songs_with_log_reg(seed_songs, data, n_recommendations=10):
   # Extract features for seed songs
   seed_song_features = [get_song_data(song['name'], data) for song in seed_songs]
   seed_song_features = np.array(seed_song_features)

   # Predict popularity for seed songs
   popularity_prob = log_reg.predict_proba(seed_song_features)[:, 1]

   # Recommend songs based on popularity prediction
   recommended_songs = data.iloc[popularity_prob.argsort()[::-1][:n_recommendations]]
   return recommended_songs

# Call the updated recommendation function
recommended_songs_with_log_reg = recommend_songs_with_log_reg(seed_songs, df)

# Print the recommended songs
for idx, song in enumerate(recommended_songs_with_log_reg.iterrows(), start=1):
   print(f"{idx}. {song[1]['name']} by {song[1]['artists']} ({song[1]['year']})")

plt.figure(figsize=(8, 6))

# Plot ROC curve for each class that has positive samples
for i in range(2):
   if np.sum(y_test == i) > 0:  # Check if there are positive samples for this class
       plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# Plot the diagonal line
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

# Set labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.show()
