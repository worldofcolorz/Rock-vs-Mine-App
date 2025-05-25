import streamlit as st
import pandas as pd
from predict import predict_object, model
from database import init_db, log_prediction, get_prediction_history

init_db()

# --- Page Config ---
st.set_page_config(page_title="Rock vs Mine Classifier", layout="wide")



st.markdown("""
<style>
/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #005f73 !important; /* Oceanic teal */
}

/* Make all text inside sidebar white */
section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: bold;
    font-family: Georgia, serif;
}

/* Custom sidebar heading style */
.sidebar-heading {
    font-size: 24px;
    font-weight: bold;
    color: #e0fbfc;
    padding-bottom: 40px;
    font-family: Georgia, serif;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<div class='sidebar-heading'>🌊 Navigation</div>", unsafe_allow_html=True)
    section = st.radio("Select Section", ["Home", "Single Prediction", "Batch Prediction"])



st.markdown(
    """
    <div style="background-color:#001f3f;padding:30px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🚢 Rock vs Mine Detection Using Sonar Signals</h1>
        <p style="color:#ccc;text-align:center;">A Machine Learning Project to Simulate Submarine Sonar Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)

if section == "Home":
    st.markdown("""
<div style="background-color:#2c3e50;padding:15px 20px;border-radius:8px;margin-top:20px;">
  <p style="color:white;font-size:18px;margin:0;padding-left:10px;">
    🔍 Use the sidebar to test single or batch predictions.
  </p>
</div>
""", unsafe_allow_html=True)


elif section == "Single Prediction":
    # Paste your single prediction form block here
    # --- Single Prediction Section ---
 with st.container():
    st.markdown("""
    <div style='background-color:#f0f8ff;padding:20px;border-radius:10px;margin-bottom:20px'>
    <h3>🔍 Single Prediction</h3>
    """, unsafe_allow_html=True)

    with st.form("single_predict_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            input_text = st.text_area("Enter 60 comma-separated sonar values", "")
        with col2:
            submitted = st.form_submit_button("Predict")

        if submitted:
            if input_text:
                result = predict_object(input_text.split(','))
                st.success(result)
                clean_result = result.split()[-1]
                log_prediction(input_text, clean_result, "Single")
            else:
                st.warning("Please enter valid input values.")

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Batch Prediction":
    # Paste your batch prediction CSV block here
    # --- Batch Prediction Section ---
 with st.container():
    st.markdown("""
    <div style='background-color:#e0f7fa;padding:20px;border-radius:10px;margin-bottom:20px'>
    <h3>📁 Batch Prediction from CSV</h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file with 60 columns (no header)", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, header=None)

            if input_df.shape[1] != 60:
                st.error("❌ CSV must have exactly 60 columns.")
            else:
                if st.button("🔍 Predict from CSV"):
                    predictions = model.predict(input_df)
                    result_df = input_df.copy()
                    result_df["Prediction"] = ["Rock" if val == 'R' else "Mine" for val in predictions]
                    st.success("✅ Predictions completed.")
                    st.dataframe(result_df)

                    for i, row in result_df.iterrows():
                        input_row = ",".join([str(val) for val in row[:-1].values])
                        log_prediction(input_row, row["Prediction"], "Batch")

                    csv_output = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Result CSV", data=csv_output, file_name="batch_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error reading file: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)


# --- Submarine Video ---
st.markdown("#### 🎥 Real-Time Submarine Sonar Demo")
st.markdown("""
<iframe width="100%" height="450" 
src="https://www.youtube.com/embed/jlXrm6gjGq8?si=O24GbRA7J6o4Hbsy&amp;start=201" 
title="YouTube video player" frameborder="0" 
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Make selected options in multiselect smaller */
div[data-baseweb="tag"] {
    font-size: 13px !important;
    padding: 4px 8px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Evaluation Report ---
with st.container():
    st.markdown("""
    <div style='background-color:#fff3e0;padding:20px;border-radius:10px;margin-bottom:20px'>
    <h3 style='color:#1e293b;'>📊 Model Evaluation Dashboard</h3>
    """, unsafe_allow_html=True)

    eval_options = st.multiselect(
        "Select Evaluation Metrics to Display:",
        ["Accuracy Score", "Classification Report", "Confusion Matrix", "Label Distribution"],
        default=["Accuracy Score"]
    )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv("Sonar data.csv", header=None)
    X = df.drop(columns=60)
    y = df[60]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

    col1, col2 = st.columns(2)

    if "Accuracy Score" in eval_options:
        with col1:
            acc = model.score(X_test, y_test)
            st.metric(label="🎯 Test Accuracy", value=f"{acc * 100:.2f}%")

    if "Classification Report" in eval_options:
        with col2:
            st.markdown("#### 📄 Classification Report")
            report_df = pd.read_csv("classification_report.csv", index_col=0)
            st.dataframe(report_df.style.background_gradient(cmap="YlGnBu"))

    if "Confusion Matrix" in eval_options:
        st.markdown("#### 🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Rock', 'Mine'], yticklabels=['Rock', 'Mine'], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

    if "Label Distribution" in eval_options:
        st.markdown("#### 📊 Label Distribution in Dataset")
        label_counts = df[60].value_counts()
        fig_dist, ax_dist = plt.subplots()
        sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax_dist)
        ax_dist.set_xlabel("Label")
        ax_dist.set_ylabel("Count")
        ax_dist.set_title("Distribution of Rock vs Mine")
        st.pyplot(fig_dist)

    st.markdown("</div>", unsafe_allow_html=True)

    


# --- Blog Section ---
    with st.expander("🧾 Read Full Project Blog"):
        st.markdown("""
        <div style='background-color:#fdf6e3;padding:20px;border-radius:10px;'>
        <h3 style='color:#2c3e50;'>📘 Project Overview – Rock vs Mine Sonar Signal Classifier</h3>

        <p style='font-size:16px;line-height:1.6;'>
        This project titled <strong>"Rock vs Mine Detection Using Sonar Signals"</strong> is a machine learning-based web application that classifies whether an object detected underwater is a rock or a mine based on sonar signal reflection data. The core idea revolves around simulating the process submarines use in identifying underwater hazards using AI.
        </p>

        <p style='font-size:16px;line-height:1.6;'>
        We used the <strong>Sonar dataset</strong> obtained from <strong>Kaggle</strong> which contains 60 numerical features per sample — each representing the energy of a sonar wave reflected off an object. The model we used is <strong>Logistic Regression</strong>, chosen for its simplicity and interpretability. Though the problem sounds straightforward, training and preprocessing this model was one of the toughest yet most rewarding parts. Dealing with real-world sonar signal noise and classifying it accurately was a challenge.
        </p>

        <p style='font-size:16px;line-height:1.6;'>
        Once trained, we built an interactive web application using <strong>Streamlit</strong>. The application allows both <strong>single predictions</strong> and <strong>batch predictions via CSV upload</strong>. One of the advanced features is that we log all predictions to a local <strong>SQLite database</strong> which can be used later to retrain or improve the model further.
        </p>

        <p style='font-size:16px;line-height:1.6;'>
        Another significant aspect of this project is the way we built it. With limited prior experience, we leaned heavily on self-learning. Thanks to <strong>OpenAI’s ChatGPT Plus Membership</strong>, we were able to break problems down and build the app step by step. The premium responses and coding guidance helped us debug and modularize our entire codebase.
        </p>

        <p style='font-size:16px;line-height:1.6;'>
        The project was created as a <strong>Major Project</strong> by our team <strong>Colorz</strong>, consisting of <strong>Ritika</strong>, <strong>Ujjwal</strong>, and <strong>Priyanshu</strong>. Under the inspiring mentorship of <strong>Priyanka Gupta Ma'am</strong>, this project came to life at <strong>BBDEC (Babu Banarasi Das Engineering College)</strong>, affiliated to <strong>AKTU</strong>. Her encouragement and faith in practical, hands-on learning were pivotal in our journey.
        </p>

        <p style='font-size:16px;line-height:1.6;'>
        Technologies used in the project include <strong>Python, Pandas, NumPy, scikit-learn, Streamlit, SQLite</strong> and <strong>Matplotlib</strong>. Key features of the app include:
        </p>

        <ul style='font-size:16px;line-height:1.6;'>
            <li>📥 CSV-based batch prediction</li>
            <li>📊 Real-time model evaluation with confusion matrix and metrics</li>
            <li>📚 Prediction history logging to local database</li>
            <li>🎨 Fully interactive and responsive UI</li>
        </ul>

        <p style='font-size:16px;line-height:1.6;'>
        In short, this project blends machine learning, web development, and real-world simulation into a simple yet impactful demonstration of how AI can assist in underwater hazard detection. We thank our college, our guide, and especially OpenAI for helping us achieve this milestone.
        </p>
                    <h4 style='color:#0a9396;'>🚀 Future Scope</h4>
                    <p style='font-size:16px;line-height:1.6;'>
                    This project lays the foundation for real-time underwater object detection using AI. In the future, we plan to:
                    <ul>
                    <li>🔁 Integrate real-time sonar signal feeds via sensors</li>
                    <li>📡 Deploy the model on embedded systems or Raspberry Pi</li>
                    <li>📊 Compare multiple models (SVM, Random Forest)</li>
                    <li>🌐 Add user login + personalized prediction dashboard</li>
                    <li>📈 Use prediction history to improve the model over time</li>
                    </ul>
                    </p>


        <p style='font-style:italic;color:#555;'>~ Team Colorz</p>

        </div>
        """, unsafe_allow_html=True)

# --- Prediction History ---
with st.container():
    st.markdown("""
    <div style='background-color:#ede7f6;padding:20px;border-radius:10px;margin-bottom:20px'>
    <h3>📜 Prediction History</h3>
    """, unsafe_allow_html=True)

    if st.button("📂 Show History"):
        history = get_prediction_history()
        if history:
            history_df = pd.DataFrame(history, columns=["ID", "Timestamp", "Input", "Result", "Method"])
            st.dataframe(history_df)
        else:
            st.info("No prediction history found.")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<style>
.custom-footer {
    background-color: #001f3f;  /* Dark navy blue */
    padding: 21px;
    border-top: 1px solid #444;
    text-align: center;
    color: white;
    font-size: 18px;
    margin-top: 50px;
    border-radius: 8px 8px 0 0;
}
</style>

<div class="custom-footer">
    © 2025 | Developed by Team Colorz (Ritika, Ujjwal, Priyanshu) | Powered by Streamlit & OpenAI ChatGPT Plus
</div>
""", unsafe_allow_html=True)

