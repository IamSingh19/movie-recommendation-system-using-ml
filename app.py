import pandas as pd
import joblib
import warnings
from warnings import filterwarnings
filterwarnings("ignore")


def load_data(file_path):
    data = pd.read_csv('movie_data_for_app.csv')
    dataframe = pd.read_csv('movie_dataframe_for_app.csv')
    return data, dataframe

def load_model(file_path):
    sig = joblib.load('sigmoid_kernel.pkl')
    tfv = joblib.load('tfidf_vectorizer.pkl')
    return sig, tfv

data, dataframe = load_data()
sig, tfv = load_model()



def give_recommendations(movie_title, model, data, dataframe):

    indices = pd.Series(data=data.index, index=data['original_title'])

    idx = indices[movie_title]

    model_scores = list(enumerate(model[idx]))

    model_scores_sorted = sorted(model_scores, key=lambda x : x[1], reverse=True)

    model_scores_10 = model_scores_sorted[1:11]

    movie_indices_10 = [i[0] for i in model_scores_10] 

    return dataframe['original_title'][movie_indices_10]




import streamlit as st

st.set_page_config(page_title="Simple Movie Recommender", layout="centered")
st.title("🎬 Simple Movie Recommender")

st.write('Find movies similar to your favorite ones!')

movie_list = data['original_title'].sort_values().tolist()

selected_movie = st.selectbox("Select a Movie : " , movie_list)

if st.button("Get Recommendations"):
    if selected_movie:
        recommendations = give_recommendations(selected_movie, sig, data, dataframe)
        
        st.subheader("Movie similar to : " + " " + selected_movie)
        for index, movie in enumerate(recommendations):
            st.write(str(index+1) + " . " + movie)
        
st.markdown("---") 
st.markdown("This app uses Content based filtering")       
