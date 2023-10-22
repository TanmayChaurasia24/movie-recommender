import streamlit as st
import pickle
import requests

st.title('movierecommendersystem')

def fetch_poster(movie_id):
    url="https://api.themoviedb.org/3/movie/{}?api_key=0b6009ac10e2ce41dc80de9c5be92e02&language=en-US".format(movie_id)
    data=requests.get(url)
    data=data.json()
    poster_path=data['poster_path']
    full_path="https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path

def recommend(moviee):
    index=movie[movie['title']==moviee].index[0]
    distances=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x:x[1])

    recommended_posters=[]
    recommended_movie=[]
    for i in distances[1:6]:
        #fetchthemovieposter
        movie_id=movie.iloc[i[0]].id
        recommended_posters.append(fetch_poster(movie_id))
        recommended_movie.append(movie.iloc[i[0]].title)
    return recommended_movie,recommended_posters

movie=pickle.load(open('movie_list.pkl','rb'))
similarity=pickle.load(open('similarity.pkl','rb'))
movie_list=movie['title'].values
name_of_selected_movie=st.selectbox('whichmovieyouwanttosee',(movie_list))

if(st.button('search')):
    recommended_movie_names,recommended_movie_posters = recommend(name_of_selected_movie)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])