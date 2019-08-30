from flask import Flask,render_template,request,url_for
import pickle
import joblib
import numpy as np 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model_list = []
with open("new_1.pickle", "rb") as f:
    for _ in range(pickle.load(f)):
        model_list.append(pickle.load(f))

ohe = model_list[0]
regressor = joblib.load('regressor.sav')

def pred_method(essay,essay_id):
    list1 = []
    list1.append(essay_id)
    sent_count = nltk.sent_tokenize(essay)
    sent_count = len(sent_count)
    list1.append(sent_count)
    #Average sentence length
    word_count = len(essay.split())
    list1.append(int(word_count/sent_count))
    
    #Cleaning the essay
    new_essay = re.sub(r'@[A-Z]{2,}\d?,?\'?s? ?','',essay)   #removing the @location words
    new_essay = re.sub(r'[^a-zA-Z ]','',new_essay)                    #removing the punctuation marks
    new_essay = nltk.word_tokenize(new_essay)
    new_essay = " ".join(new_essay)
    #char count
    char_count = len(re.sub(r'[\s]','',new_essay))
    list1.append(char_count)
    #word_count
    word_count = len(new_essay.split())
    list1.append(word_count)
    #from nltk.stem.wordnet import WordNetLemmatizer 
    #uni-_word_count
    ps = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    uniq_word = len(list(set([ps.stem(word) for word in new_essay.lower().split() if not word in stop_word])))
    list1.append(uniq_word)

    x = new_essay.lower()
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
    list1.extend([noun_count , verb_count , adjective_count, adverb_count])
    
    list1 = np.array(list1).reshape(1,-1)
    
    
    
    features_test = ohe.transform(list1).toarray()

    features_test = features_test[:,1:]

    prediction = regressor.predict(features_test)
    return prediction[0]

app = Flask(__name__, template_folder = 'templates')

@app.route('/',methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/essay01',methods=['GET','POST'])
def essay01():      
    return render_template('essay01.html')

@app.route('/essay02',methods=['GET','POST'])
def essay02():      
    return render_template('essay02.html')

@app.route('/essay03',methods=['GET','POST'])
def essay03():
    return render_template('essay03.html')

@app.route('/essay04',methods=['GET','POST'])
def essay04():      
    return render_template('essay04.html')

@app.route('/essay05',methods=['GET','POST'])
def essay05():      
    return render_template('essay05.html')

@app.route('/essay06',methods=['GET','POST'])
def essay06():      
    return render_template('essay06.html')

@app.route('/essay07',methods=['GET','POST'])
def essay07():      
    return render_template('essay07.html')

@app.route('/essay08',methods=['GET','POST'])
def essay08():      
    return render_template('essay08.html')

@app.route('/result',methods = ['GET','POST'])
def result():
    essay_id = 1
    essay = str(request.form['essay'])
    score = pred_method(essay,essay_id)
    return render_template('result.html',score=score)

@app.route('/contact_us')
def contact_us():
    return render_template('contact.html')

@app.route('/how')
def how():
    return render_template("how.html")
if __name__ == '__main__':
	app.run()

