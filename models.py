from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




def train_models(X_train, X_test, y_train, y_test):
    """Train models with scaling and cross-validation, and calculate accuracies."""
    
    # Scale features
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_train_accuracy = accuracy_score(y_train, nb_model.predict(X_train))
    nb_test_accuracy = accuracy_score(y_test, nb_model.predict(X_test))
    nb_cv_scores = cross_val_score(nb_model, X_train, y_train, cv=5, scoring='accuracy')

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_train_accuracy = accuracy_score(y_train, lr_model.predict(X_train))
    lr_test_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
    rf_test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

    return (
        nb_model, lr_model, rf_model,
        nb_train_accuracy, nb_test_accuracy,
        nb_cv_scores.mean(), nb_cv_scores.std(),
        lr_train_accuracy, lr_test_accuracy,
        lr_cv_scores.mean(), lr_cv_scores.std(),
        rf_train_accuracy, rf_test_accuracy,
        rf_cv_scores.mean(), rf_cv_scores.std()
    )
