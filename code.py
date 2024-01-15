# import libraries

# !pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
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
    new = ""
    for char in text:
        if char.isalnum():
            new += char
        else:
            new += ' '
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
    # remove poun fun
    text=remove_poun(text)
    
    # To lower fun
    text=to_lower(text)

  
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

feature_reshaped = np.array(feature).reshape(-1, 1)
feature_reshaped

label.head()

data["sentiment"].value_counts()

# Aboove we see that the data is totally imbalance now we can balance them

# Build a Pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter

# Create a dictionary for RandomUnderSampler
under_sampling_strategy = {class_label: int(0.5 * count) for class_label, count in Counter(label).items()}

under_sampling_strategy

resampling_pipeline = Pipeline([
    ('under', RandomUnderSampler(sampling_strategy=under_sampling_strategy)),  # Downsampling majority class
    ('over', RandomOverSampler(sampling_strategy='auto')),  # Upsampling minority classes
])

x_resampled, y_resampled = resampling_pipeline.fit_resample(feature_reshaped, label)

x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=42)



x_train.shape

x_train.shape

x_test.shape

y_test.shape

y_train[y_train==2].value_counts()

y_train[y_train==1].value_counts()

y_train[y_train==0].value_counts()

y_test.shape

# Now the data is balance now we perform next step

# from sklearn.pipeline import Pipeline
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



pipeline.fit(x_train,y_train)



pipe2.fit(x_train, y_train)

pipe3.fit(x_train, y_train)

# # naivebase
pre=pipeline.predict(x_test)

# # Logistic Regression
pred=pipe2.predict(x_test)

# # SVC
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

cv=cross_val_score(pipeline,x_train,y_train,cv=10)

cv.mean()

# cv2=cross_val_score(pipe2,x_train,y_train,cv=10)
# cv3=cross_val_score(pipe3,x_train,y_train,cv=10)

# print(cv2.mean(),cv3.mean())

# Pickle the model

import pickle as pkl
with open("Sentiment_analysis.pkl", "wb") as f:
    pkl.dump(pipe2, f)



# import joblib

# # Save object
# joblib.dump(pipe3, "Sentiment_analysis.pkl")

# Load the model

with open("Sentiment_analysis.pkl","rb") as f:
    m=pkl.load(f)


# test on some text

s="you are a very slow The service was slow, the food was bland, and the overall atmosphere was disappointing."
m.predict([s])


new="As the relentless storm clouds gathered overhead, a palpable sense of foreboding enveloped the once serene landscape. The biting winds carried with them the stench of impending doom, and the heavens unleashed torrents of rain, washing away any semblance of tranquility. Each raindrop seemed to echo the mournful dirge of shattered dreams. The world, once vibrant with promise, now appeared draped in a shroud of desolation. Every step forward felt like an arduous journey through a murky swamp of despair, the ground sinking beneath the weight of unmet expectations. The skeletal remains of wilted flowers mirrored the decay of optimism, and the air was thick with the acrid taste of bitter disappointment. In this disheartening tableau, the once bright horizon now loomed ominously, casting a shadow that seemed to swallow the very essence of hope."

m.predict([new])

n="The sun dipped below the horizon, casting long shadows across the quiet town. A gentle breeze rustled through the leaves, and the distant hum of crickets filled the evening air. The scent of blooming flowers mingled with the earthy aroma of damp soil, creating a serene ambiance. Streetlights flickered to life, casting a warm glow on the cobblestone streets. In this tranquil moment, time seemed to slow, and the world embraced a peaceful stillness."

m.predict([n])

d="In the heart of the metropolis, neon lights painted the night in a kaleidoscope of colors. The vibrant nightlife unfolded, with laughter and music echoing through the streets. Each corner held a story, and the city's pulse quickened as it embraced the diversity of its nocturnal inhabitants."

m.predict([d])

s="The movie was a complete letdown. The plot was confusing, the characters were poorly developed, and I left the theater feeling thoroughly disappointed."

m.predict([s])





