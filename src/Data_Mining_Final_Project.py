#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from time import time
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


import category_encoders as ce
from sklearn.metrics import classification_report, log_loss, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# In[2]:


data = pd.read_csv('music_genre-3.csv')
data


# In[3]:


data.shape


# In[4]:


data.duplicated().any()
duplicated = data.duplicated


# In[5]:


data[duplicated]


# In[6]:


data.iloc[9999:10006]


# In[7]:


data.drop([1000, 1001, 1002, 1003, 1004], inplace = True)


# In[8]:


df_music = data.dropna()


# In[9]:


df_music.isnull().any()


# In[10]:


df_music.shape


# In[11]:


len(df_music["instance_id"].unique()) 


# In[12]:


len(df_music["artist_name"].unique())


# In[13]:


df_music["key"].unique()


# In[14]:


df_music["mode"].unique()


# In[15]:


df_music["obtained_date"].unique() 


# In[16]:


df_music["music_genre"].unique()


# In[17]:


df_music.reset_index(inplace = True)


# In[18]:


df_music= df_music.drop(["index", "instance_id", "track_name", "obtained_date"], axis = 1)


# In[19]:


df_music.head()


# In[20]:


df_music[df_music["artist_name"] == "empty_field"]


# In[21]:


artists = df_music["artist_name"].value_counts()[:20].sort_values(ascending = True)
artists


# In[22]:


df_music = df_music.drop(df_music[df_music["artist_name"] == "empty_field"].index)


# In[23]:


df_music.drop("artist_name", axis = 1, inplace = True)


# In[24]:


df_music


# In[25]:


df_music[df_music["tempo"] == "?"]


# In[26]:


df_music= df_music.drop(df_music[df_music["tempo"] == "?"].index)


# In[27]:


df_music["tempo"] = df_music["tempo"].astype("float")
df_music["tempo"] = np.around(df_music["tempo"], decimals = 2)


# In[28]:


from sklearn.preprocessing import LabelEncoder
key_encoder = LabelEncoder()

df_music["key"] = key_encoder.fit_transform(df_music["key"])


# In[29]:


df_music.head()


# In[30]:


key_encoder.classes_


# In[31]:


mode_encoder = LabelEncoder()
df_music["mode"] = mode_encoder.fit_transform(df_music["mode"])
df_music.head()


# In[32]:


df_music.describe()


# In[33]:


mode_encoder.classes_


# In[34]:


music_features = df_music.drop("music_genre", axis = 1)
music_labels = df_music["music_genre"]


# In[35]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(music_features , music_labels, test_size = 0.3, random_state=42)


# In[36]:


clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = music_features.columns

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}\n")

# Feature Importance
feature_importance_dt = clf.feature_importances_

# Sort feature importance in descending order
sorted_feature_importance = sorted(zip(feature_names, feature_importance_dt), key=lambda x: x[1], reverse=True)

# Print the sorted feature importance scores
print("Feature Importance:\n")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


# In[37]:


df_dt_results = pd.DataFrame({
    'Evaluation': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})
print(df_dt_results)


# In[38]:


plotted_feature_importance_dt = pd.Series(feature_importance_dt, index = feature_names)
fig, ax = plt.subplots()
plotted_feature_importance_dt.plot.bar(ax=ax)
ax.set_title("Feature Importances for Decision Tree (CART)")
ax.set_ylabel("Feature Importance Value")
fig.tight_layout()


# In[39]:


clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters as needed

# Train the classifier on the training data
clf_rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}\n")

# Feature Importance
feature_importance_rf = clf_rf.feature_importances_

# Sort feature importance in descending order
sorted_feature_importance = sorted(zip(feature_names, feature_importance_rf), key=lambda x: x[1], reverse=True)

# Print the sorted feature importance scores
print("Feature Importance:\n")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


# In[40]:


df_dt_results = pd.DataFrame({
    'Evaluation': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})
print(df_dt_results)


# In[41]:


plotted_feature_importance_rf = pd.Series(feature_importance_rf, index = feature_names)
fig, ax = plt.subplots()
plotted_feature_importance_rf.plot.bar(ax=ax)
ax.set_title("Feature Importances for Random Forest")
ax.set_ylabel("Feature Importance Value")
fig.tight_layout()


# In[42]:


clf_stump = DecisionTreeClassifier(max_depth = 1)

# Train the classifier on the training data
clf_stump.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = music_features.columns

# Make predictions on the testing data
y_pred = clf_stump.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}\n")

# Feature Importance
feature_importance_stump = clf_stump.feature_importances_

# Sort feature importance in descending order
sorted_feature_importance = sorted(zip(feature_names, feature_importance_stump), key=lambda x: x[1], reverse=True)

# Print the sorted feature importance scores
print("Feature Importance:\n")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


# In[43]:


df_dt_results = pd.DataFrame({
    'Evaluation': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})
print(df_dt_results)


# In[44]:


plotted_feature_importance_stump = pd.Series(feature_importance_stump, index = feature_names)
fig, ax = plt.subplots()
plotted_feature_importance_stump.plot.bar(ax=ax)
ax.set_title("Feature Importances for Decision Stump")
ax.set_ylabel("Feature Importance Value")
fig.tight_layout()


# In[45]:


class_names=sorted(df_music['music_genre'].unique().tolist())
print(class_names)
M=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,M)) 
reverse_mapping=dict(zip(M,class_names))       
df_music['music_genre']=df_music['music_genre'].map(normal_mapping)

random.seed(2023)
N=list(range(len(df_music)))
random.shuffle(N)
df_music=df_music.iloc[N]


# In[46]:


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


# In[47]:


df_music=labelencoder(df_music)
display(df_music)


# In[48]:


music_features = df_music.drop("music_genre", axis = 1)
music_labels = df_music["music_genre"]


# In[49]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(music_features , music_labels, test_size = 0.3, random_state=42)


# In[50]:


clf_boost1 = xgb.XGBClassifier(objective='multi:softmax', num_class=10, random_state=42)

# Train the classifier on the training data
clf_boost1.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = music_features.columns

# Make predictions on the testing data
y_pred = clf_boost1.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}\n")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance_boost1 = clf_boost1.feature_importances_

# Sort feature importance in descending order
sorted_feature_importance = sorted(zip(feature_names, feature_importance_boost1), key=lambda x: x[1], reverse=True)

# Print the sorted feature importance scores
print("Feature Importance:\n")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


# In[51]:


df_dt_results = pd.DataFrame({
    'Evaluation': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})
print(df_dt_results)


# In[52]:


plotted_feature_importance_boost1 = pd.Series(feature_importance_boost1, index = feature_names)
fig, ax = plt.subplots()
plotted_feature_importance_boost1.plot.bar(ax=ax)
ax.set_title("Feature Importances for Decision Stump")
ax.set_ylabel("Feature Importance Value")
fig.tight_layout()


# In[53]:


# Plot feature importance
xgb.plot_importance(clf_boost1)
plt.show()


# In[54]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(music_features , music_labels , test_size=0.3, stratify = music_labels, random_state = 7)


# In[55]:


clf_boost2 = xgb.XGBClassifier(objective='multi:softmax', n_estimators = 101, random_state=42, learning_rate = 0.1)

# Train the classifier on the training data
clf_boost2.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = music_features.columns

# Make predictions on the testing data
y_pred = clf_boost1.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}\n")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance_boost2 = clf_boost2.feature_importances_

# Sort feature importance in descending order
sorted_feature_importance = sorted(zip(feature_names, feature_importance_boost2), key=lambda x: x[1], reverse=True)

# Print the sorted feature importance scores
print("Feature Importance:\n")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


# In[56]:


df_dt_results = pd.DataFrame({
    'Evaluation': ['Accuracy', 'Precision', 'Recall'],
    'Score': [accuracy, precision, recall]
})
print(df_dt_results)


# In[57]:


plotted_feature_importance_boost2 = pd.Series(feature_importance_boost2, index = feature_names)
fig, ax = plt.subplots()
plotted_feature_importance_boost2.plot.bar(ax=ax)
ax.set_title("Feature Importances for Decision Stump")
ax.set_ylabel("Feature Importance Value")
fig.tight_layout()


# In[58]:


import xgboost as xgb
import matplotlib.pyplot as plt

# Plot feature importance
xgb.plot_importance(clf_boost2)
plt.show()


# In[ ]:




