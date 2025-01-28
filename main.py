import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load CSS styles
css = """
<style>
@keyframes colorChange {
    0% { color: #ff7f50; }
    25% { color: #6495ed; }
    50% { color: #7fff00; }
    75% { color: #ff69b4; }
    100% { color: #ff7f50; }
}
.cool-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    animation: colorChange 5s infinite;
}
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
}
.stButton button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 15px 30px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 18px;
    margin: 10px 0;
    cursor: pointer;
    border-radius: 12px;
    width: 100%;
}
.stButton button:hover {
    background-color: #45a049;
}
.sidebar-title {
    font-size: 24px;
    font-weight: bold;
    color: #FFD700;
    margin-bottom: 10px;
}
.button-container {
    display: flex;
    justify-content: space-around;
    margin-top: 20px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

# Preprocess the data
def preprocess_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

# Train models and calculate accuracies
def train_models(X_train, X_test, y_train, y_test):
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

# Predict spam
def predict_spam(model, vectorizer, email):
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    probability = model.predict_proba(email_vec)[0][1]
    return prediction[0], probability

# Plot train and test accuracies
def plot_accuracies(train_accuracies, test_accuracies):
    models = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
    fig, ax = plt.subplots()
    ax.plot(models, train_accuracies, marker='o', label='Train Accuracy', color='#ff7f50')
    ax.plot(models, test_accuracies, marker='o', label='Test Accuracy', color='#6495ed')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy')
    ax.legend()
    st.pyplot(fig)

# Streamlit App
def main():
    st.markdown('<div class="cool-title">Spam Email Classifier</div>', unsafe_allow_html=True)

    # Sidebar content
    st.sidebar.markdown("<div class='sidebar-title'>Project Information</div>", unsafe_allow_html=True)
    st.sidebar.write("**Student**: Arash Niroumand")
    st.sidebar.write("**Course**: Artificial Intelligence")
    st.sidebar.write("This project demonstrates the use of machine learning models to classify emails as spam or not spam.")
    st.sidebar.write("**Models Used**:")
    st.sidebar.write("- Naive Bayes")
    st.sidebar.write("- Logistic Regression")
    st.sidebar.write("- Random Forest")

    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)
    (
        nb_model, lr_model, rf_model,
        nb_train_accuracy, nb_test_accuracy,
        lr_train_accuracy, lr_test_accuracy,
        rf_train_accuracy, rf_test_accuracy
    ) = train_models(X_train, X_test, y_train, y_test)

    # Session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "Models Info"

    # Navigation buttons in the main page
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Models Info"):
            st.session_state.page = "Models Info"
    with col2:
        if st.button("Check Emails"):
            st.session_state.page = "Check Emails"
    with col3:
        if st.button("Diagrams"):
            st.session_state.page = "Diagrams"
    st.markdown("</div>", unsafe_allow_html=True)

    # Display selected page
    if st.session_state.page == "Models Info":
        st.subheader("Model Accuracies")
        st.write(f"Naive Bayes Train Accuracy: {nb_train_accuracy * 100:.2f}%")
        st.write(f"Naive Bayes Test Accuracy: {nb_test_accuracy * 100:.2f}%")
        st.write(f"Logistic Regression Train Accuracy: {lr_train_accuracy * 100:.2f}%")
        st.write(f"Logistic Regression Test Accuracy: {lr_test_accuracy * 100:.2f}%")
        st.write(f"Random Forest Train Accuracy: {rf_train_accuracy * 100:.2f}%")
        st.write(f"Random Forest Test Accuracy: {rf_test_accuracy * 100:.2f}%")

    elif st.session_state.page == "Check Emails":
        st.subheader("Test the Model")
        model_option = st.selectbox("Select a model:", ["Naive Bayes", "Logistic Regression", "Random Forest"])
        email_input = st.text_area("Enter an email to check if it's spam:")

        if st.button("Predict"):
            if email_input:
                if model_option == "Naive Bayes":
                    model = nb_model
                elif model_option == "Logistic Regression":
                    model = lr_model
                else:
                    model = rf_model

                prediction, probability = predict_spam(model, vectorizer, email_input)

                if prediction == 1:
                    st.error(f"This email is **SPAM** with {probability * 100:.2f}% probability.")
                else:
                    st.success(f"This email is **NOT SPAM** with {(1 - probability) * 100:.2f}% probability.")
            else:
                st.warning("Please enter an email to check.")

    elif st.session_state.page == "Diagrams":
        st.subheader("Train vs Test Accuracy")
        train_accuracies = [nb_train_accuracy, lr_train_accuracy, rf_train_accuracy]
        test_accuracies = [nb_test_accuracy, lr_test_accuracy, rf_test_accuracy]
        plot_accuracies(train_accuracies, test_accuracies)

# Run the app
if __name__ == "__main__":
    main()