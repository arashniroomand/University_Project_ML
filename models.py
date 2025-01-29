from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def train_models(X_train, X_test, y_train, y_test):
    """Train models and calculate accuracy with cross-validation."""

    # 1️⃣ Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_cv_mean = cross_val_score(nb_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    nb_test_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

    # 2️⃣ Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_cv_mean = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    lr_test_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

    # 3️⃣ Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_cv_mean = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    rf_test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

    return (
        nb_model, lr_model, rf_model,
        nb_cv_mean, nb_test_accuracy,
        lr_cv_mean, lr_test_accuracy,
        rf_cv_mean, rf_test_accuracy
    )
