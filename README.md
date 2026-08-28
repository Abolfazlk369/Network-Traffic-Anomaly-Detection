# 🚀 Network Traffic Forecasting & Anomaly Detection using PyTorch & LSTM

سیستم پیش‌بینی هوشمند ترافیک شبکه و کشف ناهنجاری (آنومالی / حملات سایبری) با استفاده از یادگیری عمیق و شبکه‌های عصبی LSTM در PyTorch.

An Intelligent Network Traffic Forecasting and Anomaly Detection System built using PyTorch, LSTM Neural Networks, and Scikit-Learn.

---

## 📌 درباره پروژه (About The Project)

در این پروژه، داده‌های زمانی ترافیک شبکه (بر حسب Mbps) به همراه الگوهای واقعی روزانه، آخر هفته‌ها و اسپایک‌های ناگهانی (شبیه‌ساز حملات DDoS یا آنومالی) شبیه‌سازی شده‌اند. سپس یک مدل یادگیری عمیق **LSTM** برای یادگیری الگوی ترافیک آموزش داده شده و با محاسبه میزان خطای پیش‌بینی (Residuals)، رفتارهای غیرعادی و حملات شبکه شناسایی می‌شوند.

This project simulates complex network traffic time-series data featuring daily cycles, weekend drops, and sudden traffic spikes (representing DDoS attacks or anomalies). A Deep Learning **LSTM** architecture is used to learn sequence patterns and accurately flag network anomalies based on prediction error thresholds.

---

## ✨ ویژگی‌های کلیدی (Key Features)

* **شبیه‌سازی واقع‌گرایانه داده‌ها (Data Simulation):** ترکیب چرخه‌های روزانه (Sinusoidal)، الگوهای آخر هفته و حملات ناگهانی DDoS.
* **پیش‌پردازش و پنجره زمانی (Time-Series Windowing):** آماده‌سازی داده‌ها با روش Lookback Window جهت پردازش سری‌های زمانی.
* **مدل یادگیری عمیق LSTM:** معماری ۲ لایه LSTM به همراه Dropout برای جلوگیری از Overfitting و لایه‌های Dense با PyTorch.
* **سیستم کشف آنومالی (Anomaly Detection):** محاسبه آستانه بحرانی بر اساس انحراف معیار خطای پیش‌بینی ($Mean + 3 \times Std$).
* **تجسم‌سازی متحرک و نمودار (Visualization):** رسم نمودار مقایسه‌ای ترافیک واقعی، پیش‌بینی مدل و علامت‌گذاری آنومالی‌ها با Matplotlib.

---

## 🛠 پیش‌نیازها و نصب (Requirements & Installation)

برای اجرای پروژه نیازمند پایتون ۳.۸ به بالا و کتابخانه‌های زیر هستید:

```bash
pip install numpy pandas matplotlib torch scikit-learn
