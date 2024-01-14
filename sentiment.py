import pickle as pkl
import joblib
import string as s
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

port =PorterStemmer()

# Convert to lower case
def to_lower(text):
    return text.lower()

# Remove pouncation
def remove_poun(text):
    new=""
    for i in text:
        if(i.isalnum()):
            new+=i
        else:
            new+=" "
    return new

# remove stop words
def remove_stopword(text):
    stopword=set(stopwords.words("english"))
    
    new_text=[word for word in text.split() if word not in stopword]
    
    return " ".join(new_text)

# Stem
def stem(text):
    return port.stem(text)

# Tokenize 
def tokenize(text):
    return word_tokenize(text)
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

with open("new.pkl","rb") as f:
    model=pkl.load(f)
s="i am a good boy"
print(model.predict([s]))