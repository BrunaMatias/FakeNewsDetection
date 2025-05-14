from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X_test):
        return self.vectorizer.transform(X_test)
