import streamlit as st
import pickle as pkl
# from pymongo import MongoClient
# Load the model necessary library
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

# Connecting to mongodb
# client = MongoClient("mongodb://localhost:27017")
# db = client["Sentimentdb"]
# collection = db["User_Sentiment"]


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


def full_process(text):
    process_text=[]
    for i in text.split("."):
        new_text=text_process(i)
        process_text.append(new_text)

    return " ".join(process_text)


# load the model
def load_model():
    with open("Sentiment_analysis.pkl","rb") as f:
        model=pkl.load(f)
    return model

# Make a gui
# Home page
def home():
    st.title("Sentiment Anlysis")
    st.write("welome to our sentiment analysis page in this page you will privide a text either positive or negative our machine learning model will tell you your text is positive or negative or neutral.")

     # Display an image
    st.markdown(
        f'<img src="https://source.unsplash.com/400x300/?movies" alt="Movies" style="height:250px; padding-left:100px;">',
        unsafe_allow_html=True
    )
    # Take a input
    st.markdown(
        f'<h4>would you like image or not enter your review in some line.</h4>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<h3>Enter review below: </h3>',
        unsafe_allow_html=True
    )
    user_input=st.text_area("",height=50)

    # place the button and take the user input
    if st.button("Sentiment Check"):
        # Set the output box
        #  st.text_area("Your output",user_input)
        # process the user input
        final_text=full_process(user_input)

        # load the model

        model=load_model()

        # predict

        pre=model.predict([final_text])
            
        if pre[0] == 0:
            sentiment_color = "red"
            sentiment_text = "Negative"
        elif pre[0] == 1:
            sentiment_color = "yellow"
            sentiment_text = "Neutral"
        else:
            sentiment_color = "green"
            sentiment_text = "Positive"

        # Insert document
        response_data = {
        "user_input": user_input,
        "predicted_sentiment": sentiment_text
        }
        collection.insert_one(response_data)
            
        # Display the sentiment with different colors
        if(sentiment_color=="green" and sentiment_text=="Positive"):
            st.success(f"your text sentiment: {sentiment_text}")
            st.markdown(   
                f'<font color="{sentiment_color}">{sentiment_text}: {user_input}</font>',
                unsafe_allow_html=True,
            )
        elif(sentiment_color=="red" and sentiment_text=="Negative"):
            st.error(f"your text sentiment: {sentiment_text}")
            st.markdown(   
                f'<font color="{sentiment_color}">{sentiment_text}: {user_input}</font>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"your text sentiment: {sentiment_text}")
            st.markdown(   
                f'<font color="{sentiment_color}">{sentiment_text}: {user_input}</font>',
                unsafe_allow_html=True,
            )
    else:
        st.text_area("Your output")
        
    
# Showing the code both model and production
def Check_code():
    st.title("Code Page")
    st.write("Source code of both Model Development and  Production")
  
    st.markdown(
        f"<a href='https://github.com/Sami606713' >View Github</a>",
        unsafe_allow_html=True

    )

    if(st.button("Model Code")):
        with open("code.py") as f:
            content=f.read()
        st.code(content,language="python")
    if(st.button("Web Code")):
        with open("app.py") as f:
            content=f.read()
        st.code(content,language="python")
    

# Main fun 
def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Code"]
    selected_page = st.sidebar.radio("Go to", pages)

    # Display the selected page
    if selected_page == "Home":
        home()
    elif selected_page == "Code":
        Check_code()
        
if __name__=="__main__":
    main()