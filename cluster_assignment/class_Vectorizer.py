from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class CustomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, vectorizer_type='tfidf', ngram_range=(1, 1),max_df=0.8,min_df=0.01):
        self.stop_words = stop_words
        self.vectorizer_type = vectorizer_type
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df

    def fit(self, X, y=None):
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range)
        else:
            self.vectorizer = CountVectorizer(stop_words=self.stop_words, ngram_range=self.ngram_range)
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


if __name__ == "__main__":
    # Create a vectorizer that uses TF-IDF
    vectorizer_tfidf = CustomVectorizer(stop_words='english', vectorizer_type='tfidf')
    # Create a vectorizer that uses CountVectorizer
    vectorizer_count = CustomVectorizer(stop_words='english', vectorizer_type='count')
