import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

def load_data(file_path='output.csv'):
    """
    Load and preprocess the dataset from a local CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        df (pd.DataFrame): DataFrame containing the loaded data with 'label' and 'text' columns.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Display the DataFrame (for debugging purposes)
    print(df)
    
    # Map labels to binary values (spam: 1, ham: 0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data by splitting into training and testing sets and vectorizing the text.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train_vec (sparse matrix): Vectorized training data.
        X_test_vec (sparse matrix): Vectorized testing data.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    # Ensure no missing or NaN values in the text column
    df = df.dropna(subset=['text'])  # Drop rows where 'text' is NaN
    df['text'] = df['text'].astype(str)  # Ensure all text data is string type
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
     
    # Print class distribution in training and testing sets
    print("Class distribution in training data:")
    print(y_train.value_counts())
    print("\nClass distribution in testing data:")
    print(y_test.value_counts())
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer
