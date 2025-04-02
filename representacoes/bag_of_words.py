from sklearn.feature_extraction.text import CountVectorizer

class BagOfWords:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X_test):
        return self.vectorizer.transform(X_test)
