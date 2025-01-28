import matplotlib.pyplot as plt
import streamlit as st

def predict_spam(model, vectorizer, email):
    """Predict whether an email is spam."""
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    probability = model.predict_proba(email_vec)[0][1]
    return prediction[0], probability

def plot_accuracies(train_accuracies, test_accuracies):
    """Plot train vs test accuracies."""
    models = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
    fig, ax = plt.subplots()
    ax.plot(models, train_accuracies, marker='o', label='Train Accuracy', color='#ff7f50')
    ax.plot(models, test_accuracies, marker='o', label='Test Accuracy', color='#6495ed')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy')
    ax.legend()
    st.pyplot(fig)