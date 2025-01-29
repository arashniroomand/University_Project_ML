import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
def load_data(file_path):
    """Load and preprocess the dataset."""
    url = file_path
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def preprocess_data(df):
    """Split and vectorize the data."""
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer