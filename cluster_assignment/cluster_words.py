from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, MiniBatchNMF,NMF
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

#############################################  Preprocessing ###############################################
def vectorizer_tfidf(texts, stopwords, ngram_range=(1, 1), max_df=0.8, min_df=0.01):
    """
    Create a TF-IDF Document-Term Matrix from texts.
    """
    vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    data_tfidf_noun = vectorizer.fit_transform(texts)
    data_dtm_noun = pd.DataFrame(data_tfidf_noun.toarray(), columns=vectorizer.get_feature_names_out())
    data_dtm_noun.index = range(len(texts))  # Adjust index to match your data
    return data_dtm_noun, vectorizer

def vectorizer_count(texts, stopwords, ngram_range=(1, 1), max_df=0.8, min_df=0.01):
    """
    Create a Count Document-Term Matrix from texts.
    """
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    data_count_noun = vectorizer.fit_transform(texts)
    data_dtm_noun = pd.DataFrame(data_count_noun.toarray(), columns=vectorizer.get_feature_names_out())
    data_dtm_noun.index = range(len(texts))  # Adjust index to match your data
    return data_dtm_noun, vectorizer

#############################################  Modeling ###############################################
def modeling_nmf(data_tfidf, vectorizer, n_components=5, max_iter=1000, alpha_W=0.00005, alpha_H=0.00005):
    X = data_tfidf.values
    nmf_model = NMF(n_components=n_components, max_iter=max_iter,  alpha_W=alpha_W, alpha_H=alpha_H, random_state=42)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(H):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return nmf_model

def modeling_minibatch_nmf(data_noun,vectorizer,n_components=5, max_iter=1000,alpha_W=0.00005, alpha_H=0.00005):
    X = data_noun.values
    minibatchnmf_model = MiniBatchNMF(n_components=n_components, max_iter=max_iter, alpha_W=alpha_W, alpha_H=alpha_H, random_state=42)
    W = minibatchnmf_model.fit_transform(X)
    H = minibatchnmf_model.components_
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(H):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return minibatchnmf_model

def modeling_lda(data_noun,vectorizer,n_components=5, max_iter=1000):
    X = data_noun.values
    lda_model = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, random_state=42)
    W = lda_model.fit_transform(X)
    H = lda_model.components_
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(H):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return lda_model

############################################# Visualize ############################################
def visualize_topic_wordcloud(model, feature_names, n_components):
    """
    Plot word clouds for each topic.
    Parameters:
    - model: Fitted topic model (e.g., NMF, LDA).
    - feature_names: List of feature names from the vectorizer.
    - n_components: Number of topics.
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, n_components * 2))
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        topic_words = {feature_names[i]: topic[i] for i in topic.argsort()[:-20:-1]}  
        wordcloud = WordCloud(
            background_color='white',
            max_words=20,
            colormap='viridis'
        ).generate_from_frequencies(topic_words)
        ax = axes[topic_idx]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_top_words(model, feature_names, n_top_words, n_components, title):
    fig, axes = plt.subplots(1, 5, figsize=(12, 6), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 10})
        ax.tick_params(axis="both", which="major", labelsize=10)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=12)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# ############################################ Evaluate ############################################
def evaluate_model(data_noun,model):
    W = model.fit_transform(data_noun.values)
    H = model.components_
    print("Silhouette Score:", silhouette_score(W, model.transform(data_noun.values).argmax(axis=1)))

def evaluate_model_lda(data_noun,lda_model):
    """ EVALUATE MODEL BY (LDA Perplexity"""
    W = lda_model.fit_transform(data_noun.values)
    H = lda_model.components_
    print("Perplexity Score:", lda_model.perplexity(data_noun.values))
