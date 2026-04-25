# Fake News Detection Project

A machine learning-powered web application designed to detect fake news, specifically tailored for Indian news contexts, Hinglish, and social media forwards.

## 🚀 Features

- **User Authentication**: Secure Register and Login system using SQLite and password hashing.
- **Fake News Prediction**: Real-time classification of news text into **FAKE NEWS**, **TRUE NEWS**, or **OTHER NEWS** (for low-confidence or non-news content).
- **ML Engine**: Uses TF-IDF vectorization and a Logistic Regression model.
- **Admin Logging**: Automatically logs all user registrations to an Excel file (`data/users.xlsx`).
- **Modern UI**: Clean, responsive frontend built with HTML, CSS (Poppins font), and JavaScript.
- **Step-by-Step Flow**: Guided user experience (Register → Login → Predict).

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, Joblib
- **Database**: SQLite (User Auth)
- **Logging**: Openpyxl (Excel)
- **Frontend**: HTML5, CSS3 (Flexbox/Gradients), JavaScript (Fetch API)

## 📁 Project Structure

```text
vaibhav-project/
├── app.py              # Main Flask application & Auth logic
├── train_model.py      # ML training script (TF-IDF + Logistic Regression)
├── data/
│   ├── true.csv        # Dataset: Verified true news
│   ├── fake.csv        # Dataset: Known fake news/forwards
│   ├── users.db        # SQLite database for user accounts
│   └── users.xlsx      # Excel log of registered users
├── model/
│   └── fake_news_model.pkl  # Trained ML model pipeline
├── static/
│   ├── css/style.css   # Custom styles & animations
│   └── js/app.js       # Frontend prediction logic
└── templates/
    ├── index.html      # Prediction page
    ├── login.html      # User login page
    └── register.html   # User registration page
```

## ⚙️ Installation & Setup

1. **Clone or Open the Project**:
   Ensure you are in the project root directory.

2. **Install Dependencies**:
   ```bash
   pip install flask scikit-learn pandas joblib openpyxl
   ```

3. **Train the Model**:
   Run the training script to generate the `.pkl` model file.
   ```bash
   python train_model.py
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Web App**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## 📖 Usage Guide

1. **Register**: Create a new account with a username and password.
2. **Login**: Sign in with your registered credentials.
3. **Analyze**: Paste any news headline or WhatsApp forward into the text area.
4. **Result**: Click "Analyze" to see if the news is likely **FAKE** or **TRUE**, along with a confidence score.

## 🛡️ Future Enhancements
- Integration with live News APIs (NewsAPI, etc.).
- Real SMS/Email OTP verification (hooks already prepared in code).
- Advanced Deep Learning models (LSTM/Transformers).

---
© 2026 Fake News Detector • India
