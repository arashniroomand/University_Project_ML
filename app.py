import streamlit as st
from data_loader import load_data, preprocess_data
from models import train_models
from utils import predict_spam, plot_accuracies
import pandas as pd

# Load CSS styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Function to display an image
def display_image():
    image_url = "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzI5d2xxbTI5YjNmcXU0Mno2emk2cGI4NDlsdHhsYmNvN2s5azE1diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WsuHnF9xWZqz9HbjtO/giphy.gif"
    st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="{image_url}" alt="N-Puzzle Image" style="max-width: 100%; width: 250px; height: auto;">'
        f'</div>',
        unsafe_allow_html=True,
    )

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
        nb_cv_mean, nb_cv_std,
        lr_train_accuracy, lr_test_accuracy,
        lr_cv_mean, lr_cv_std,
        rf_train_accuracy, rf_test_accuracy,
        rf_cv_mean, rf_cv_std
    ) = train_models(X_train, X_test, y_train, y_test)

    # Session state for navigation
    if "show_image" not in st.session_state:
        st.session_state.page = "Models Info"
        st.session_state.show_image = True  # Show image by default

    # Navigation buttons in the main page
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Models Info"):
            st.session_state.page = "Models Info"
            st.session_state.show_image = False  # Hide image when a button is clicked
    with col2:
        if st.button("Check Emails"):
            st.session_state.page = "Check Emails"
            st.session_state.show_image = True
    with col3:
        if st.button("Diagrams"):
            st.session_state.page = "Diagrams"
            st.session_state.show_image = False  # Hide image when a button is clicked
    st.markdown("</div>", unsafe_allow_html=True)

    # Display image by default
    if st.session_state.show_image:
        display_image()

    # Page: Models Info
    if st.session_state.page == "Models Info" and not st.session_state.show_image:
        st.subheader("Model Accuracies and Cross-Validation Metrics")

        # Create a DataFrame to organize the accuracies and CV metrics
        data = {
            "Model": ["Naive Bayes", "Logistic Regression", "Random Forest"],
            "Train Accuracy (%)": [nb_train_accuracy * 100, lr_train_accuracy * 100, rf_train_accuracy * 100],
            "Test Accuracy (%)": [nb_test_accuracy * 100, lr_test_accuracy * 100, rf_test_accuracy * 100],
            "CV Mean (%)": [nb_cv_mean * 100, lr_cv_mean * 100, rf_cv_mean * 100],
            "CV Std Dev (%)": [nb_cv_std * 100, lr_cv_std * 100, rf_cv_std * 100],
        }
        df = pd.DataFrame(data)

        # Display the table
        st.table(df)

    # Page: Check Emails
    elif st.session_state.page == "Check Emails":
        st.session_state.show_image = True
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

    # Page: Diagrams
    elif st.session_state.page == "Diagrams" and not st.session_state.show_image:
        st.subheader("Train vs Test Accuracy")
        train_accuracies = [nb_train_accuracy, lr_train_accuracy, rf_train_accuracy]
        test_accuracies = [nb_test_accuracy, lr_test_accuracy, rf_test_accuracy]
        plot_accuracies(train_accuracies, test_accuracies)

# Run the app
if __name__ == "__main__":
    main()
