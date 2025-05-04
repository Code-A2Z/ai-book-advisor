## AI Based Book Recommendation System
***
### 🎯 **Goal**

The main goal of this project is to develop an AI-based book recommendation system that can suggest similar books to users based on a given book title. The purpose is to enhance the book discovery experience by leveraging deep learning and hybrid recommendation techniques.

### 🧵 **Dataset**

Dataset link: https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/

### 🎥 **YouTube**

[![Image](https://github.com/user-attachments/assets/1a93c606-434f-4928-84a5-de0e9b72f2b3)](https://www.youtube.com/watch?v=Vf8WjgzIF1U)

### 🧾 **Description**

This project aims to improve upon an existing book recommendation system by integrating an AI model that suggests books from a large dataset. The recommendation system combines traditional content-based filtering with neural collaborative filtering (NCF) and convolutional neural network (CNN) models to enhance recommendation accuracy.

### 🧮 **What I had done!**

1. **Data Preprocessing**:
   - Cleaned and prepared the dataset by handling missing values and encoding categorical data.
   - Generated TF-IDF vectors for book descriptions and computed cosine similarity for content-based filtering.

2. **Model Training**:
   - Implemented and trained a Neural Collaborative Filtering (NCF) model using user and book embeddings.
   - Implemented and trained a Convolutional Neural Network (CNN) model to predict book ratings based on descriptions.

3. **Hybrid Recommendation System**:
   - Combined the results from cosine similarity, NCF, and CNN models to provide a robust recommendation system.

4. **Deployment**:
   - Created a Streamlit application to provide an interface for users to get book recommendations.

### 🚀 **Models Implemented**

1. **Cosine Similarity**: Used for content-based filtering based on book descriptions.
   - Chosen for its simplicity and effectiveness in measuring similarity between text vectors.

2. **Neural Collaborative Filtering (NCF)**: Used for collaborative filtering based on user-book interactions.
   - Chosen for its ability to learn user and item embeddings that capture latent factors in the data.

3. **Convolutional Neural Network (CNN)**: Used to predict book ratings based on textual descriptions.
   - Chosen for its ability to capture complex patterns in text data through hierarchical feature extraction.

### 📚 **Libraries Needed**

- **pandas**
- **numpy**
- **scikit-learn**
- **torch**
- **tensorflow**
- **keras**

### 📈 **Performance of the Models based on the Accuracy Scores**

- **Neural Collaborative Filtering (NCF) Model**:
  - Test MAE: 0.2378

- **Convolutional Neural Network (CNN) Model**:
  - Test MAE: 0.2390

- **Cosine Similarity**:
  - Provides top-N recommendations based on content similarity.
