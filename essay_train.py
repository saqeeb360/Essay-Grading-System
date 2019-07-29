# -*- coding: utf-8 -*-
"""Essay_score.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Kehkm9f9JNZWWEp82LoQFDyHkYVAd5x1

author :
Saqeeb, SKIT college-ECE branch
Shamim Banu, SKIT college-ECE branch

"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer

#reading the data in pandas dataframe with encoding - charmap and delimiter '\t'
df = pd.read_csv("training_set_rel3.tsv", delimiter = "\t",encoding='charmap')
#new_df = pd.read_csv('new_df.csv',encoding='charmap')
df = df.iloc[:,[0,1,2,6]]

'''
Counting the number of sentences in each essay

'''

def sent_count(x):
    essay = str(x)
    sent_list = nltk.sent_tokenize(essay)
    return len(sent_list)
df['sent_count'] = df['essay'].apply(sent_count)


"""
Average sentence length

"""

def av_sent_len(x):
    essay = str(x)
    word_count = len(essay.split())
    sent_list = nltk.sent_tokenize(str(x))
    sent_count = len(sent_list)
    return int(word_count/sent_count)
df['av_sent_len'] = df['essay'].apply(av_sent_len)

"""
Cleaning the essay

"""

def clean_essay(x):
  x = re.sub(r'@[A-Z]{2,}\d?,?\'?s? ?','',str(x))
  x = re.sub(r'[^a-zA-Z ]','',x)
  x = nltk.word_tokenize(x)
  x = " ".join(x)
  return x
df['essay'] = df['essay'].apply(clean_essay)

"""
Adding char count
"""

def char_count(x):
  x = re.sub(r'[\s]','',str(x))
  return len(x)           
df['char_count'] = df['essay'].apply(char_count)

"""
Adding word count

"""

def word_count(x):
  essay = str(x)
  essay = essay.split()
  return len(essay)    
df['word_count'] = df['essay'].apply(word_count)


"""
Number of distinct words in a essay

"""

ps = PorterStemmer()
stop_word = set(stopwords.words('english'))
def uni_word_count(x):
    essay = str(x).lower()
    uniq_word = essay.split()
    uniq_word = [ps.stem(word) for word in uniq_word if not word in stop_word]
    uniq_word = list(set(essay.split()))   
    return len(uniq_word)
df['uni_word_count'] = df['essay'].apply(uni_word_count)


"""
Nouns_count , Verb_count , Adjective_count, Adverb_count

Can also use textblob and spacy
"""
def lemmas_count(x):
  x = str(x).lower()
  x = nltk.word_tokenize(x)
  pos_tag = nltk.pos_tag(x)
  
  noun_count = 0
  verb_count = 0
  adjective_count = 0
  adverb_count = 0
  
  for (word, pos) in pos_tag:
    if pos.startswith('N'):
      noun_count +=1
    elif pos.startswith('V'):
      verb_count +=1
    elif pos.startswith('J'):
      adjective_count += 1
    elif pos.startswith('R'):
      adverb_count +=1
  return pd.Series([noun_count , verb_count , adjective_count, adverb_count])
df[['noun_count','verb_count','adjective_count','adverb_count']] = df['essay'].apply(lemmas_count)


df1 = df[df['essay_set'] == 1]
df2 = df[df['essay_set'] == 2]
df3 = df[df['essay_set'] == 3]
df4 = df[df['essay_set'] == 4]
df5 = df[df['essay_set'] == 5]
df6 = df[df['essay_set'] == 6]
df7 = df[df['essay_set'] == 7]
df8 = df[df['essay_set'] == 8]


"""
Making a ndarray for training set
Features and label
Also we need to tokenize the df

"""


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000, stop_words='english')
features = cv.fit_transform(df['essay'])
features = features.toarray()

temp = df.iloc[:,[1,4,5,6,7,8,9,10,11,12]].values
features = np.hstack((temp , features))


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features= [0])
features = onehotencoder.fit_transform(features).toarray()
features = features[:,1:]

features = features.astype('int32')

# Preparing the labels
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
labels_1 = df1.iloc[:, [3]].values
labels_1= sc1.fit_transform(labels_1)

sc2 = StandardScaler()
labels_2 = df2.iloc[:, [3]].values
labels_2= sc2.fit_transform(labels_2)

sc3 = StandardScaler()
labels_3 = df3.iloc[:, [3]].values
labels_3= sc3.fit_transform(labels_3)

sc4 = StandardScaler()
labels_4= df4.iloc[:, [3]].values
labels_4= sc4.fit_transform(labels_4)

sc5 = StandardScaler()
labels_5 = df5.iloc[:, [3]].values
labels_5= sc5.fit_transform(labels_5)

sc6 = StandardScaler()
labels_6 = df6.iloc[:, [3]].values
labels_6= sc6.fit_transform(labels_6)

sc7 = StandardScaler()
labels_7 = df7.iloc[:, [3]].values
labels_7= sc7.fit_transform(labels_7)

sc8 = StandardScaler()
labels_8 = df8.iloc[:, [3]].values
labels_8= sc8.fit_transform(labels_8)

labels = np.concatenate((labels_1,labels_2,labels_3,labels_4,labels_5,labels_6,labels_7,labels_8))








import matplotlib.pyplot as plt
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,4])
plt.title('Sentence Count')
plt.show()  
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,5])
plt.title('Average Sentence Count')
plt.show()  
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,6])
plt.title('Character Count')
plt.show()  
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,7])
plt.title('Word Count')
plt.show()  
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,8])
plt.title('Distinct Word Count')
plt.show()  
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,9])
plt.title('Noun Count')
plt.show()  
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,10])
plt.title('Verb Count')
plt.show()
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,11])
plt.title('Adjective Count')
plt.show()
plt.scatter(x = df.iloc[:1783,3], y=df.iloc[:1783,12])
plt.title('Adverb Count')
plt.show()

"""
Training from the data
"""
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=25, random_state=0)  
regressor.fit(features, labels)  

score = regressor.score(features,labels)
print(score)



'''
Training the model with Deep Learning using Artificial neural network(ANN)

Importing the Keras libraries and packages

'''
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#adding the first hidden layer
classifier.add(Dense(units = 400, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1016))

# Adding the second hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error' )

classifier.summary()

classifier.fit(features , labels,epochs = 50 , batch_size = 506)


# Making a file of the trained model and objects to reuse it for predictions.

obj_features = []
obj_features.append(cv)              #Bag of words - count vectorizers
obj_features.append(onehotencoder)   #Essay set onrhotencoder
obj_features.extend([sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8]) #Standard scaler of each essay set.


import pickle 
with open('new_1.pickle', "wb") as f:
  pickle.dump(len(obj_features), f)
  for value in obj_features:
      pickle.dump(value, f)

import joblib
joblib.dump(regressor,'regressor.sav')






'''
classifier.save("classifier.h5")

from keras.models import load_model
classifier = load_model("classifier.h5")
import pickle
model_list = []
with open('new_1.pickle', "rb") as f:
    for _ in range(pickle.load(f)):
        model_list.append(pickle.load(f))
'''

'''
from sklearn.externals import joblib
with open('joblib.sav', "wb") as f:
  joblib.dump(len(obj_features), f)
  for value in obj_features:
      joblib.dump(value, f)
model_list = []
with open('joblib.sav', "rb") as f:
    for _ in range(joblib.load(f)):
        model_list.append(joblib.load(f))






model_list[3]

model_pred = model_list[3].predict(features_test)

model_pred1 = sc1.inverse_transform(model_pred[:valid_list[0]])
model_pred2 = sc2.inverse_transform(model_pred[valid_list[0]:valid_list[1]])
model_pred3 = sc3.inverse_transform(model_pred[valid_list[1]:valid_list[2]])
model_pred4 = sc4.inverse_transform(model_pred[valid_list[2]:valid_list[3]])
model_pred5 = sc5.inverse_transform(model_pred[valid_list[3]:valid_list[4]])
model_pred6 = sc6.inverse_transform(model_pred[valid_list[4]:valid_list[5]])
model_pred7 = sc7.inverse_transform(model_pred[valid_list[5]:valid_list[6]])
model_pred8 = sc8.inverse_transform(model_pred[valid_list[6]:valid_list[7]])

model_pred = np.concatenate((model_pred1,model_pred2,model_pred3,model_pred4,model_pred5,model_pred6,model_pred7,model_pred8),axis = 0)

model_pred = model_pred.astype('int64')

model_pred[:10] , labels_pred[:10]

mode_pred = modelmodel_preddf_test = 0
valid_score = 0

df_test = pd.read_csv('valid_set.tsv' , delimiter = '\t' , encoding = 'charmap')
valid_score = pd.read_csv('valid_sample_submission_5_column.csv' , encoding = 'charmap')

df_test.head(),valid_score.head()

len(df_test['essay_id']),len(valid_score['essay_id'])

list1 = []
def present(x):
  if int(x) not in list1:
    list1.append(int(x))
    return 1
  else:
    return np.NaN
valid_score['present'] = valid_score['essay_id'].apply(present)

valid_score['present'].unique()

valid_score = valid_score.dropna(axis = 0)

df_test.shape , valid_score.shape

df_test.head(2)

df_test = df_test.iloc[:,[0,1,2]]

df_test['sent_count'] = df_test['essay'].apply(sent_count)

df_test['av_sent_len'] = df_test['essay'].apply(av_sent_len)

df_test['essay'] = df_test['essay'].apply(clean_essay)

df_test['word_count'] = df_test['essay'].apply(word_count)

df_test['uni_word_count'] = df_test['essay'].apply(uni_word_count)

df_test['noun_count'] = df_test['essay'].apply(noun_count)

df_test['verb_count'] = df_test['essay'].apply(verb_count)

df1_test = df_test[df_test['essay_set'] == 1]
df2_test = df_test[df_test['essay_set'] == 2]
df3_test = df_test[df_test['essay_set'] == 3]
df4_test = df_test[df_test['essay_set'] == 4]

df5_test = df_test[df_test['essay_set'] == 5]
df6_test = df_test[df_test['essay_set'] == 6]
df7_test = df_test[df_test['essay_set'] == 7]
df8_test = df_test[df_test['essay_set'] == 8]

valid_score1 = valid_score[valid_score['essay_set'] == 1]
valid_score2 = valid_score[valid_score['essay_set'] == 2]
valid_score3 = valid_score[valid_score['essay_set'] == 3]
valid_score4 = valid_score[valid_score['essay_set'] == 4]

valid_score5 = valid_score[valid_score['essay_set'] == 5]
valid_score6 = valid_score[valid_score['essay_set'] == 6]
valid_score7 = valid_score[valid_score['essay_set'] == 7]
valid_score8 = valid_score[valid_score['essay_set'] == 8]

df1_test.shape ,df2_test.shape ,df3_test.shape ,df4_test.shape ,df5_test.shape ,df6_test.shape ,df7_test.shape ,df8_test.shape

valid_shape = [valid_score1.shape,valid_score2.shape,valid_score3.shape,valid_score4.shape,valid_score5.shape,valid_score6.shape,valid_score7.shape,valid_score8.shape]

valid_shape



features_test = cv.transform(df_test['essay'])
features_test = features_test.toarray()

features_test.shape

df_test.head(2)

temp_test = df_test.iloc[:,[1,3,5,6,7,8]].values
features_test = np.hstack((temp_test , features_test))

features_test.shape , temp_test.shape

features_test.shape

features_test = onehotencoder.transform(features_test).toarray()
features_test.shape

features_test = features_test[:,1:]
features_test.shape

features_test = sc.transform(features_test)

features_test.shape

labels_pred = classifier.predict(features_test)

len(labels_pred)

valid_shape



valid_list = []
sum = 0
for i in valid_shape:
  sum = sum + i[0]
  valid_list.append(sum)

valid_list

labels_pred1 = sc1.inverse_transform(labels_pred[:valid_list[0]])
labels_pred2 = sc2.inverse_transform(labels_pred[valid_list[0]:valid_list[1]])
labels_pred3 = sc3.inverse_transform(labels_pred[valid_list[1]:valid_list[2]])
labels_pred4 = sc4.inverse_transform(labels_pred[valid_list[2]:valid_list[3]])
labels_pred5 = sc5.inverse_transform(labels_pred[valid_list[3]:valid_list[4]])
labels_pred6 = sc6.inverse_transform(labels_pred[valid_list[4]:valid_list[5]])
labels_pred7 = sc7.inverse_transform(labels_pred[valid_list[5]:valid_list[6]])
labels_pred8 = sc8.inverse_transform(labels_pred[valid_list[6]:valid_list[7]])

labels_pred = np.concatenate((labels_pred1,labels_pred2,labels_pred3,labels_pred4,labels_pred5,labels_pred6,labels_pred7,labels_pred8),axis = 0)

labels_pred.shape

labels_pred = labels_pred.astype('int32')

labels_pred[:10],valid_score1['predicted_score'][:10]



plt.figure(figsize=[16,7])
plt.title("Comparision of actual and predicted score")
plt.plot(df_test['essay_id'][valid_list[6]:valid_list[7]],labels_pred[valid_list[6]:valid_list[7]], color = 'red',label = 'Predicted score')
plt.plot(df_test['essay_id'][valid_list[6]:valid_list[7]],valid_score['predicted_score'][valid_list[6]:valid_list[7]] , color = 'blue' , label = 'Actual score')
plt.ylabel("Score")
plt.xlabel("Essay_id")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=[16,7])
plt.title("Comparision of actual and predicted score")
plt.plot(df_test['essay_id'][:] , labels_pred[:], color = 'red',label = 'Predicted score')
plt.plot(df_test['essay_id'][:] , valid_score['predicted_score'][:] , color = 'blue' , label = 'Actual score')
plt.ylabel("Score")
plt.xlabel("Essay_id")
plt.legend()
plt.grid()
plt.show()

'''