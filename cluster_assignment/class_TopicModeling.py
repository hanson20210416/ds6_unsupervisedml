from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation
from sklearn.base import BaseEstimator, TransformerMixin
from class_Vectorizer import CustomVectorizer

class TopicModeling(BaseEstimator, TransformerMixin):
    def __init__(self, model_type='nmf', n_components=5, max_iter=1000, alpha_W=0.00005, alpha_H=0.00005):
        self.model_type = model_type
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H

    def fit(self, X, y=None):
        if self.model_type == 'nmf':
            self.model = NMF(n_components=self.n_components, max_iter=self.max_iter, 
                             alpha_W=self.alpha_W, alpha_H=self.alpha_H, random_state=42)
        elif self.model_type == 'minibatch_nmf':
            self.model = MiniBatchNMF(n_components=self.n_components, max_iter=self.max_iter, 
                                      alpha_W=self.alpha_W, alpha_H=self.alpha_H, random_state=42)
        elif self.model_type == 'lda':
            self.model = LatentDirichletAllocation(n_components=self.n_components, max_iter=self.max_iter, 
                                                   random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose from 'nmf', 'minibatch_nmf', or 'lda'")
        
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_
        return self

    def transform(self, X):
        return self.model.transform(X)

    def print_topics(self, vectorizer, top_n=10):
        """
        Print the top N words for each topic.
        """
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.H):
            top_words = [feature_names[i] for i in topic.argsort()[:-top_n-1:-1]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

if __name__ == "__main__":
    data_dtm_noun_tfidf = CustomVectorizer(stop_words='english', vectorizer_type='tfidf')
    # Create a TopicModeling object for NMF
    topic_model_nmf = TopicModeling(model_type='nmf', n_components=5, max_iter=1000, alpha_W=0.00005, alpha_H=0.00005)
    # Fit the model to your data
    topic_model_nmf.fit(data_dtm_noun_tfidf)
    # Get the topics
    topics_nmf = topic_model_nmf.get_topics(vectorizer)

    # Create a TopicModeling object for MiniBatchNMF
    topic_model_minibatch = TopicModeling(model_type='minibatch_nmf', n_components=5, max_iter=1000, alpha_W=0.00005, alpha_H=0.00005)
    topic_model_minibatch.fit(data_dtm_noun_tfidf)
    topics_minibatch = topic_model_minibatch.get_topics(vectorizer)

    # Create a TopicModeling object for LDA
    topic_model_lda = TopicModeling(model_type='lda', n_components=5, max_iter=1000)
    topic_model_lda.fit(data_dtm_noun_tfidf)
    topics_lda = topic_model_lda.get_topics(vectorizer)

    # Print topics for each model
    print("NMF Topics:")
    print("\n".join(topics_nmf))
    print("#" * 100)

    print("MiniBatchNMF Topics:")
    print("\n".join(topics_minibatch))
    print("#" * 100)

    print("LDA Topics:")
    print("\n".join(topics_lda))