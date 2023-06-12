import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

@st.cache
def load_data():
    movie = pd.read_csv("movie.csv")[:5000]
    return movie

def main():
    movie = load_data()

    C = movie['Score'].mean()
    m = movie['Review count'].quantile(0.9)
    movie_list = movie.copy().loc[movie['Review count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['Review count']
        R = x['Score']
        return round(((R * v + C * m) / (v + m)), 2)

    movie_list['WR_score'] = movie_list.apply(weighted_rating, axis=1)
    movie_list = movie_list.sort_values('WR_score', ascending=False)

    movie['General'] = movie['Genre'] + ' ' + movie['Original Language'] + ' ' + movie['Rating'] + ' ' + movie['Director'] + ' ' + movie['Cast'] + ' ' + movie['Synopsis']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie['General'])

    svd = TruncatedSVD(n_components=100)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    cosine_sim = cosine_similarity(svd_matrix, svd_matrix)

    movie = movie.reset_index()
    indices = pd.Series(movie.index, index=movie['Title'])

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = movie[movie["Title"] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:30]
        movie_indices = [i[0] for i in sim_scores]
        movie_pred = movie.iloc[movie_indices][['Title', 'Genre', 'Score', 'Review count', 'Original Language']]
        C = movie['Review count'].mean()
        m = movie['Review count'].quantile(0.40)
        t = movie['Score'].quantile(0.5)
        qualified = movie_pred[(movie_pred['Review count'] >= m) & (movie_pred['Score'] >= t)]
        qualified['WR_score'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('WR_score', ascending=False).head(15)
        return qualified

    st.set_page_config(
        page_title="RECOMMEND SYSTEM",
        page_icon="🔥",
        layout="wide")
    st.title('**:blue[FILM RECOMMENDATION SYSTEM]**')

    with st.form("Thông tin"):
        options = movie["Title"]
        name = st.selectbox('**:red[Typing the film title]**', options=options)
        submit = st.form_submit_button("**Get films**")

    if submit:
        with st.spinner("Loading..."):
            time.sleep(0.25)
    try:
        a = get_recommendations(name)
        st.success('**Success**', icon="✅")
        st.write('**:orange[Here are movies similar to]**', name)
        for i in range(len(a)):
            if a.iloc[i, 0] == name:
                continue
            with st.form('' + str(i) + ''):
                st.markdown(f':green[**👀Title**:] {a.iloc[i, 0]}')
                st.markdown(f':green[**👀Genre**:] {a.iloc[i, 1]}')
                st.markdown(f':green[**👀Film score**:] {a.iloc[i, 2]}')
                st.markdown(f':green[**👀Reviews count**:] 🍅 {a.iloc[i, 3]} 🍅')
                st.markdown(f':green[**👀Original Language**:] {a.iloc[i, 4]}')
                st.markdown(f':green[**👀	WR_score**:] {a.iloc[i, 5]}')
                submit = st.form_submit_button(str(i + 1), disabled=True)
    except:
        st.error('There are no movies that are similar to ' + name, icon="❌")
