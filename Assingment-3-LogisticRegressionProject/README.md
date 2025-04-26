
# 🎯 Diabetes Prediction App using Logistic Regression

Welcome to the **Diabetes Prediction App**, a machine learning project built to predict whether a patient has diabetes based on the Pima Indians Diabetes Dataset. This app leverages a Logistic Regression model with an accuracy of ~72% and is deployed using Streamlit. 🚀

![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.12-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42.2-orange.svg)

---

## 📋 Project Overview

This project uses the **Pima Indians Diabetes Database** from Kaggle to train a Logistic Regression model. The app allows users to input patient features (e.g., Glucose, BMI, Age) and get a prediction along with probability scores. 🌡️

- **Dataset**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Model**: Logistic Regression (~72% accuracy)
- **Deployment**: Streamlit Cloud ([Live App](https://diabetes-prediction-by-lr.streamlit.app/))
- **GitHub**: [thilak-r/diabetes-prediction-using-LR](https://github.com/thilak-r/diabetes-prediction-using-LR)

---

## 🚀 Features

- 📊 Predict diabetes status (Diabetic or Not Diabetic) based on user input.
- 📈 View probability scores for both classes.
- 🎨 Optional visualization of the dataset (scatter plot of Glucose vs BMI).
- 🌐 Fully deployed and accessible online.

---

## 📂 Folder Structure

```plaintext
diabetes-classifier-streamlit/
├── app.py                # Streamlit app code
├── diabetes_model.pkl    # Trained Logistic Regression model
├── diabetes.csv          # Dataset for visualization (optional)
├── requirements.txt      # Dependency list
└── README.md             # This file
```

---

## 🛠️ Setup Instructions

### Prerequisites
- 🐍 Python 3.12 or higher
- 📦 Git (for cloning the repository)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/thilak-r/diabetes-prediction-using-LR.git
   cd diabetes-classifier-streamlit
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app locally:
   ```bash
   streamlit run app.py
   ```
   - Open `http://localhost:8501` in your browser.

---

## 🎮 Usage

1. 🖱️ Adjust the sliders in the sidebar to input patient features (e.g., Pregnancies, Glucose, BMI).
2. 🔍 Click the "Predict" button to see the result.
3. 📉 Check the "Show Dataset Visualization" box (if `diabetes.csv` is present) to view a scatter plot.

**Example Output:**
- Prediction: "Diabetic" or "Not Diabetic"
- Probabilities: e.g., "Probability of Not Diabetic: 0.65", "Probability of Diabetic: 0.35"

---

## 🌐 Deployment

The app is deployed on **Streamlit Community Cloud**:
- 🔗 [Live App](https://diabetes-prediction-by-lr.streamlit.app/)
- Deployed from the `main` branch of this repository.

To redeploy:
1. Push changes to the `main` branch.
2. Streamlit Cloud auto-updates the app.

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests! Suggestions to improve accuracy (e.g., using Random Forest) or add features are welcome. 📝

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a Pull Request.

---

## ⚠️ License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details. 📜

---

## 🙏 Acknowledgments

- 🌟 Dataset from [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) via Kaggle.
- 🎉 Streamlit Community for the free deployment platform.
- 💻 Inspiration from Dr. Agughasi Victor I.'s "Mathematics for Machine Learning" guide.

---

## 📬 Contact

- 👤 **Author**: Thilak R
- 📧 **Email**: [thilak22005@gmail.com](mailto:thilak22005@egmail.com)
- 🌐 **GitHub**: [thilak-r](https://github.com/thilak-r)


  <br><br>
under guidance of [Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu)


