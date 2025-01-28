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