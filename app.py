import streamlit as st
from data_loader import load_data, preprocess_data
from models import train_models
from utils import predict_spam, plot_accuracies

# Load CSS styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit App
def main():
    st.markdown('<div class="cool-title">Spam SMS Classifier</div>', unsafe_allow_html=True)

    # Sidebar content
    st.sidebar.markdown("<div class='sidebar-title'>Project Information</div>", unsafe_allow_html=True)
    st.sidebar.write("**Student**: Arash Niroumand")
    st.sidebar.write("**Course**: Artificial Intelligence")
    st.sidebar.write("This project demonstrates the use of machine learning models to classify SMS as spam or not spam.")
    st.sidebar.write("**Models Used**:")
    st.sidebar.write("- Naive Bayes")
    st.sidebar.write("- Logistic Regression")
    st.sidebar.write("- Random Forest")
    st.sidebar.image("img/robot.gif")

    # Load data
    file_path = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    (
        nb_model, lr_model, rf_model,
        nb_cv_mean, nb_test_accuracy,
        lr_cv_mean, lr_test_accuracy,
        rf_cv_mean, rf_test_accuracy
    ) = train_models(X_train, X_test, y_train, y_test)

    # Session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "HOME"

    # Navigation buttons in the main page
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Models Info"):
            st.session_state.page = "Models Info"
    with col2:
        if st.button("Check Emails"):
            st.session_state.page = "Check Emails"
    with col3:
        if st.button("Diagrams"):
            st.session_state.page = "Diagrams"
    with col4:
        if st.button("HOME"):
            st.session_state.page = "HOME"
    st.markdown("</div>", unsafe_allow_html=True)

    # Display selected page
    if st.session_state.page == "Models Info":
        st.subheader("Model Accuracies")
        st.write(f"Naive Bayes CV Accuracy: {nb_cv_mean * 100:.2f}%")
        st.write(f"Naive Bayes Test Accuracy: {nb_test_accuracy * 100:.2f}%")
        st.write(f"Logistic Regression CV Accuracy: {lr_cv_mean * 100:.2f}%")
        st.write(f"Logistic Regression Test Accuracy: {lr_test_accuracy * 100:.2f}%")
        st.write(f"Random Forest CV Accuracy: {rf_cv_mean * 100:.2f}%")
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
        train_accuracies = [nb_cv_mean, lr_cv_mean, rf_cv_mean]
        test_accuracies = [nb_test_accuracy, lr_test_accuracy, rf_test_accuracy]
        plot_accuracies(train_accuracies, test_accuracies)

    elif st.session_state.page == "HOME":
        st.markdown("""
        <div class="home-section">
            <h1 class="home-title">Welcome to the Spam SMS Classifier</h1>
            <p class="home-text">
                This project leverages the power of machine learning to classify SMS messages as spam or not spam. 
                Below, you'll find a brief overview of the models used in this project:
            </p>
            <div class="model-section">
                <h2 class="model-title">Naive Bayes</h2>
                <p class="model-text">
                    Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes independence between features, 
                    making it fast and efficient for text classification tasks like spam detection.
                </p>
            </div>
            <div class="model-section">
                <h2 class="model-title">Logistic Regression</h2>
                <p class="model-text">
                    Logistic Regression is a linear model that predicts the probability of a binary outcome. It's widely used 
                    for classification tasks due to its simplicity and interpretability.
                </p>
            </div>
            <div class="model-section">
                <h2 class="model-title">Random Forest</h2>
                <p class="model-text">
                    Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy 
                    and reduce overfitting. It's robust and performs well on a variety of datasets.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()