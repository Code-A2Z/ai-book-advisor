import streamlit as st
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch, pandas as pd, tensorflow as tf
import numpy as np

def recommend_books_cosine(book_title, final_data, cosine_sim, no_of_recommendations):
    if not final_data.empty:
        idx = final_data[final_data['Title'] == book_title].index
        if len(idx) > 0:
            idx = idx[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:no_of_recommendations + 1]
            book_indices = [i[0] for i in sim_scores]
            return final_data[['Title', 'Image', 'Author']].iloc[book_indices]
        else:
            return "Book not found"
    else:
        return "No data available"

def hybrid_recommendation(book_title, df, cosine_sim, ncf_model, cnn_model, tokenizer, no_of_recommendations, max_len=200):
    # Cosine Sim Reco
    cosine_recs = recommend_books_cosine(book_title, df, cosine_sim, no_of_recommendations)

    user_id = df['user_id'].iloc[0]
    book_ids = [df[df['Title'] == title]['ISBN'].values[0] for title in cosine_recs['Title']]

    # NCF Reco
    ncf_recs = []

    for book_id in book_ids:
        user_tensor = torch.tensor([int(user_id)], dtype=torch.long)
        book_tensor = torch.tensor([int(book_id)], dtype=torch.long)
        rating_pred = ncf_model([user_tensor, book_tensor])[0].numpy()
        rating_pred = tf.reshape(ncf_model([user_tensor, book_tensor]), [-1]).numpy()
        ncf_recs.append((book_id, rating_pred))

    ncf_recs = sorted(ncf_recs, key=lambda x: x[1], reverse=True)

    # CNN Reco
    cnn_recs = []
    for book_id in book_ids:
        book_desc = df[df['ISBN'] == book_id]['Desc'].values[0]
        book_seq = tokenizer.texts_to_sequences([book_desc])
        book_pad = pad_sequences(book_seq, maxlen=max_len)
        rating_pred = cnn_model.predict(book_pad).item()
        cnn_recs.append((book_id, rating_pred))

    cnn_recs = sorted(cnn_recs, key=lambda x: x[1], reverse=True)

    # Combine and Rank Reco
    combined_recs = list(set([book for book, _ in cnn_recs]))
    final_recs = [(df[df['ISBN'] == book_id]['Title'].values[0],
                  df[df['ISBN'] == book_id]['Author'].values[0],
                  df[df['ISBN'] == book_id]['Genre'].values[0])
                 for book_id in combined_recs]

    return [final_recs[i][0] for i in range(len(final_recs))]

@st.cache_resource
def Load_Model_Weights():
    gdown.download(f'https://drive.google.com/uc?id={st.secrets['COSINE_SIM']}', 'cosine_sim.npy', quiet=False)
    gdown.download(f'https://drive.google.com/uc?id={st.secrets['NCF_MODEL']}', 'ncf_model.h5', quiet=False)
    gdown.download(f'https://drive.google.com/uc?id={st.secrets['CNN_MODEL']}', 'cnn_model.h5', quiet=False)
    gdown.download(f'https://drive.google.com/uc?id={st.secrets['FINAL_DATA_WITH_RATINGS']}', 'final_data_with_ratings.csv', quiet=False)

    cosine_sim = np.load('cosine_sim.npy')
    ncf_model = load_model('ncf_model.h5')
    cnn_model = load_model('cnn_model.h5')
    df = pd.read_csv('final_data_with_ratings.csv')

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['Desc'])

    return df, cosine_sim, ncf_model, cnn_model, tokenizer

def display_book(book_data):
    with st.expander(f"📖 {book_data['Title']}", expanded=False):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(book_data['Image'], width=200)

        with col2:
            st.markdown(f"#### 📚 Title: **{book_data['Title']}**")

            authors = list(set([author.strip() for author in book_data['Author'].split(',')]))
            author_names = ' '.join(f'`{author.strip()}`' for author in authors)
            st.markdown(f"##### ✍️ Author: **{author_names}**")

            st.markdown(f"**📖 Description:** {book_data['Desc']}")

            genres = list(set([genre.strip() for genre in book_data['Genre'].split(',')]))
            genre_tags = ' '.join(f'`{genre.strip()}`' for genre in genres)
            st.markdown(f"**🏷️ Genre:** {genre_tags}")

            st.markdown(f"**📄 Pages:** {book_data['Pages']}")
            st.markdown(f"**⭐ Rating:** {book_data['Rating']}")

def app():
    st.title("💡 Book Brain: AI Book Advisor")
    st.write("This project is a hybrid book recommendation system that combines content-based filtering and collaborative filtering techniques. It uses a neural collaborative filtering model and a CNN model for text classification to provide personalized book recommendations.")

    df, cosine_sim, ncf_model, cnn_model, tokenizer = Load_Model_Weights()

    book_title = st.sidebar.selectbox("Select a book title", df['Title'].unique())
    no_of_recommendations = st.sidebar.slider("Number of recommendations", min_value=1, max_value=20, value=5)

    if st.sidebar.button("Recommend"):
        recommendations = hybrid_recommendation(book_title, df, cosine_sim, ncf_model, cnn_model, tokenizer, no_of_recommendations)
        recommendations = sorted(recommendations, key=lambda rec: df[df['Title'] == rec]['Rating'].values[0], reverse=True)

        st.markdown("### 📔 Your Book")
        selected_book_data = df[df['Title'] == book_title].iloc[0]
        display_book(selected_book_data)

        st.divider()
        st.markdown("### 📚 Similar Books")

        for rec in recommendations:
            book_data = df[df['Title'] == rec].iloc[0]
            display_book(book_data)
    
    else:
        st.image('https://media.glamour.com/photos/5fc5597957cf5666837ad199/master/w_2560%2Cc_limit/GL-MeTimeBooks-Lede.jpg')
        st.divider()
        st.markdown(
            """
            <div style="text-align: center; font-size: 20px; font-weight: bold;">
                Developed by: <span style="color: #888;">Avdhesh Varshney</span><br>
                <span style="font-size: 16px; color: #777;">Empowering your reading journey</span>
            </div>
            """,
            unsafe_allow_html=True
        )

app()
