from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_models(X_train, X_test, y_train, y_test):
    """Train models and calculate accuracies."""
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_train_accuracy = accuracy_score(y_train, nb_model.predict(X_train))
    nb_test_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_train_accuracy = accuracy_score(y_train, lr_model.predict(X_train))
    lr_test_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
    rf_test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

    return (
        nb_model, lr_model, rf_model,
        nb_train_accuracy, nb_test_accuracy,
        lr_train_accuracy, lr_test_accuracy,
        rf_train_accuracy, rf_test_accuracy
    )