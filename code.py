# import libraries
# !pip install --upgrade scikit-learn

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("full_review.csv")

data.head()

# Set the index

data.set_index("index",inplace=True)

# null values

data.isnull().sum()

data.dropna(inplace=True)

data.duplicated().sum()

data.drop_duplicates(inplace=True)

data.shape

### Add a new col that can hold the text that the review is positive or negative

from nltk.sentiment import SentimentIntensityAnalyzer

# Make a object of sentiment class
sia=SentimentIntensityAnalyzer()

# Apply the Sentiment obj
data["sentiment"]=data["review"].apply(lambda x : sia.polarity_scores(x)['compound'])

# now our data look like this

data.head()

# Now convert the sentimnt

def sentiment(nbr):
    if(nbr>0):
        return "positive"
    elif(nbr<0):
        return "negative"
    else:
        return "neutral"



# Apply the fun

data["sentiment"]=data["sentiment"].apply(sentiment)

data.head(10)

data["sentiment"].value_counts()

# Now perform text analysis

# Convert lower case
# Remove pouncation
# remove stop words
# Stem the word

# Lower case

def to_lower(text):
    return text.lower()

to_lower("sSAMi")

# Remove pouncation



import string as s
def remove_poun(text):
    new=""
    for i in text:
        if(i.isalnum()):
            new+=i
        else:
            new+=" "
    return new

n="sami!$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ullah"
remove_poun(n)

# Remove Stopword

from nltk.corpus import stopwords
def remove_stopword(text):
    stopword=set(stopwords.words("english"))
    
    new_text=[word for word in text.split() if word not in stopword]
    
    return " ".join(new_text)

remove_stopword("i am a a biy a snf i ama a good bot")

# Stem the word

from nltk.stem import PorterStemmer

port =PorterStemmer()

port.stem("I am a boy")

def stem(text):
    return port.stem(text)

stem("I am a boy")

# Tokenize the word

from nltk.tokenize import word_tokenize

def tokenize(text):
    return word_tokenize(text)

tokenize("samiullah is a good boy")

# Now all the fun in a single fun

# Now combne all the fun in a single fun
def text_process(text):
    # To lower fun
    text=to_lower(text)

    # remove poun fun
    text=remove_poun(text)
  
    # remove stop word fun
    text=remove_stopword(text)
    
    # Stem Fun
    text=stem(text)
    
    # Tokenize fun
    final=tokenize(text)
    
    return " ".join(final)

text_process("i am a good boy and you")

# data["review"]=data["review"].apply(text_process)

# Encode the label

from sklearn.preprocessing import LabelEncoder

encode=LabelEncoder()
data["sentiment"]=encode.fit_transform(data["sentiment"])







# Split the text

from sklearn.model_selection import train_test_split

feature=data["review"]
label=data['sentiment']

label.head()

data["sentiment"].value_counts()

x_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=42)

x_train.shape

y_train.shape

# Build a Pipeline

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

model=MultinomialNB()

# Multinomianl Naivebase

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=text_process,max_features=1000)),
    ('classifier', MultinomialNB())
])

# logistic Regression

pipe2 = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=text_process,max_features=1000)),
    ('classifier', LogisticRegression())
])

# SVM

pipe3 = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=text_process,max_features=1000)),
    ('classifier', SVC())
])

# Random Forest

pipe_random = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=text_process,max_features=1000)),
    ('classifier', RandomForestClassifier())
])



pipeline.fit(x_train,y_train)



pipe2.fit(x_train, y_train)

pipe3.fit(x_train, y_train)

# pipe_random.fit(x_train, y_train)





# naivebase
pre=pipeline.predict(x_test)

# Logistic Regression
pred=pipe2.predict(x_test)

# SVC
svc_pred=pipe3.predict(x_test)

# Accuracy

from sklearn.metrics import accuracy_score,r2_score

# Naivebase
accuracy=accuracy_score(y_test,pre)

# Logistic Regression
acc=accuracy_score(y_test,pred)

# SVC
svc_acc=accuracy_score(y_test,svc_pred)

print("Nib base Accuracy: ",accuracy)
print("Logistic Accuracy: ",acc)
print("svc Accuracy: ",svc_acc)



from sklearn.model_selection import cross_val_score

# cv=cross_val_score(pipeline,x_train,y_train,cv=10)

# cv.mean()

# Pickle the model

import pickle as pkl
with open("Sentiment_analysis.pkl", "wb") as f:
    pkl.dump(pipe2, f)



# Load the model

with open("Sentiment_analysis.pkl","rb") as f:
    m=pkl.load(f)

# test on some text

s="you are a very slow The service was slow, the food was bland, and the overall atmosphere was disappointing."
m.predict([s])

new="I absolutely loved the new movie! The storyline was captivating, the acting was superb, and the cinematography was breathtaking."
m.predict([new])

n="The restaurant experience was terrible. The service was slow, the food was bland, and the overall atmosphere was disappointing."

m.predict([n])

d="The seminar covered various topics related to the industry and industry area are very dirty. The speakers presented their findings, and attendees had the opportunity to ask questions during the Q&A session."

m.predict([d])

s="you are a very slow The service was slow, the food was bland, and the overall atmosphere was disappointing."

m.predict([s])





