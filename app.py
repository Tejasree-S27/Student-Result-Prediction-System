import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# DATASET GENERATION
# =====================================================

np.random.seed(42)

n = 500

hours_study = np.random.randint(1, 13, n)
attendance = np.random.randint(40, 101, n)
previous_marks = np.random.randint(30, 101, n)
assignments_done = np.random.randint(0, 11, n)
sleep_hours = np.random.randint(4, 11, n)

score = (
    hours_study * 4
    + attendance * 0.3
    + previous_marks * 0.4
    + assignments_done * 2
    + sleep_hours
)

result = np.where(score >= 70, 1, 0)

df = pd.DataFrame({
    "Hours_Study": hours_study,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Assignments_Done": assignments_done,
    "Sleep_Hours": sleep_hours,
    "Result": result
})

# =====================================================
# MODEL TRAINING
# =====================================================

X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎓 Student Result Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Data Analysis",
        "Model Performance",
        "About Project"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("🎓 Student Result Prediction System")

    st.markdown("""
    ### Welcome

    This Machine Learning application predicts whether
    a student will PASS or FAIL using:

    - Study Hours
    - Attendance
    - Previous Marks
    - Assignments Completed
    - Sleep Hours

    ### Technologies Used

    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    """)

    st.success("Machine Learning Model Ready ✅")

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("🔮 Predict Student Result")

    col1, col2 = st.columns(2)

    with col1:

        study = st.slider(
            "Study Hours",
            1,
            12,
            5
        )

        attendance_value = st.slider(
            "Attendance %",
            40,
            100,
            75
        )

        marks = st.slider(
            "Previous Marks",
            30,
            100,
            60
        )

    with col2:

        assignments = st.slider(
            "Assignments Done",
            0,
            10,
            5
        )

        sleep = st.slider(
            "Sleep Hours",
            4,
            10,
            7
        )

    if st.button("Predict Result"):

        input_data = scaler.transform([[
            study,
            attendance_value,
            marks,
            assignments,
            sleep
        ]])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("PASS ✅")

            st.metric(
                "Confidence",
                f"{probability[1]*100:.2f}%"
            )

        else:

            st.error("FAIL ❌")

            st.metric(
                "Confidence",
                f"{probability[0]*100:.2f}%"
            )

        result_df = pd.DataFrame({
            "Feature": [
                "Study Hours",
                "Attendance",
                "Previous Marks",
                "Assignments",
                "Sleep Hours"
            ],
            "Value": [
                study,
                attendance_value,
                marks,
                assignments,
                sleep
            ]
        })

        st.table(result_df)

# =====================================================
# DATA ANALYSIS PAGE
# =====================================================

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Statistics")

    st.write(df.describe())

    st.subheader("Study Hours vs Previous Marks")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Hours_Study"],
        df["Previous_Marks"]
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Previous Marks")

    st.pyplot(fig)

    st.subheader("Attendance vs Previous Marks")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Attendance"],
        df["Previous_Marks"]
    )

    ax.set_xlabel("Attendance")
    ax.set_ylabel("Previous Marks")

    st.pyplot(fig)

    st.subheader("Pass / Fail Distribution")

    fig, ax = plt.subplots(figsize=(6, 6))

    df["Result"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")

    st.pyplot(fig)

    st.subheader("Marks Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["Previous_Marks"],
        bins=10
    )

    st.pyplot(fig)

# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )

    with col2:
        st.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )

        st.metric(
            "F1 Score",
            f"{f1*100:.2f}%"
        )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    st.pyplot(fig)

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    })

    importance = importance.sort_values(
        by="Coefficient",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        importance["Feature"],
        importance["Coefficient"]
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About Project":

    st.title("About Project")

    st.markdown("""
    ### Student Result Prediction System

    This project predicts whether a student will
    PASS or FAIL using Machine Learning.

    ### Model
    Logistic Regression

    ### Dataset
    500 Synthetic Student Records

    ### Features
    - Study Hours
    - Attendance
    - Previous Marks
    - Assignments Completed
    - Sleep Hours

    ### Developed Using
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    """)