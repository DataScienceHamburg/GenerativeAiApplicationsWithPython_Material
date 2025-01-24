# Streamlit app
#%% packages
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings 

#%% load the vector database
embedding_function = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", collection_name="movies", embedding_function=embedding_function)
#%% develop the app
st.title("Movie Finder")

# Add a slider for minimum IMDB rating
min_rating = st.slider("Minimum IMDB Rating", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
# Add a single-select input for genres
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 
          'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Short', 
          'Sport', 'Thriller', 'War', 'Western']
selected_genre = st.selectbox("Select a genre", genres)



user_query = st.chat_input("What happens in the movie?")
if user_query:
    # Retrieve the most similar movies
    with st.spinner("Searching for similar movies..."):
        metadata_filter = {"imdb_rating": {"$gte": min_rating}}
        similar_movies = db.similarity_search_with_score(user_query, k=100, filter=metadata_filter)
        # filter for selected genre
        similar_movies = [movie for movie in similar_movies if selected_genre in movie[0].metadata['genres']]
        # Print the titles of the movies
        
    # Display the results
    st.header(f"Most Similar Movies: ")
    st.subheader(f"Query: '{user_query}'")
    cols = st.columns(4)
    # Check if there are duplicate results
    unique_results = []
    seen_titles = set()
    
    for doc, score in similar_movies:
        if doc.metadata['title'] not in seen_titles:
            unique_results.append((doc, score))
            seen_titles.add(doc.metadata['title'])
    
    # Display only unique results
    for i, (doc, score) in enumerate(unique_results):
        if i >= len(cols):
            break
        with cols[i % 4]:
            if doc.metadata['poster']:
                try:
                    st.image(doc.metadata['poster'], width=150)
                except:
                    st.write("No poster available")
            else:
                st.write("No poster available")
            st.markdown(f"**{doc.metadata['title']}**")
            st.write(f"Genres: {doc.metadata['genres']}")
            st.write(f"IMDB Rating: {doc.metadata['imdb_rating']}")
            st.write(f"Similarity Score: {score:.4f}")
    
    if len(unique_results) < len(similar_movies):
        st.warning(f"Note: {len(similar_movies) - len(unique_results)} duplicate result(s) were removed.")

