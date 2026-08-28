import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ۱. تنظیم دانه تصادفی جهت تکرارپذیری
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# ۲. شبیه‌سازی داده‌های پیشرفته شبکه (شامل الگوهای روزانه، هفتگی و حملات DDoS)
# ---------------------------------------------------------
print("در حال تولید داده‌های پیچیده ترافیک شبکه...")
dates = pd.date_range(start="2026-01-01", periods=24 * 60, freq="h")  # داده‌های ۶۰ روز (ساعتی)

# اصلاح اصلی اینجاست: تبدیل Indexهای پانداس به NumPy Array با استفاده از to_numpy()
daily_cycle = 400 * np.sin(np.pi * (dates.hour.to_numpy() - 6) / 12)
weekly_cycle = np.where(dates.dayofweek.to_numpy() >= 5, -150, 50)  # کاهش ترافیک در آخر هفته
base_traffic = 700
noise = np.random.normal(0, 30, len(dates))

traffic_data = base_traffic + daily_cycle + weekly_cycle + noise

# حالا traffic_data حتماً یک آرایه NumPy است و بدون خطا تغییر می‌کند
traffic_data[500:505] += 800   # اسپایک شدید ۱
traffic_data[1200:1203] += 1000 # اسپایک شدید ۲

df = pd.DataFrame({"Traffic_Mbps": traffic_data}, index=dates)

# ---------------------------------------------------------
# ۳. آماده‌سازی و نرمال‌سازی داده‌ها
# ---------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Traffic_Mbps']])


# ایجاد پنجره‌های زمانی (Lookback Window)
def create_dataset(dataset, look_back=24):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


LOOK_BACK = 24
X, y = create_dataset(scaled_data, LOOK_BACK)

# تبدیل داده‌ها به تینسورهای PyTorch (PyTorch Tensors)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # ساختار: [samples, seq_len, features]
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# تقسیم به داده‌های آموزش (۸۰٪) و تست (۲۰٪)
train_size = int(len(X_tensor) * 0.8)

X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# ساخت DataLoader برای پردازش دسته‌ای (Batch Processing)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ---------------------------------------------------------
# ۴. تعریف معماری شبکه عصبی LSTM با PyTorch
# ---------------------------------------------------------
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # انتخاب آخرین گام زمانی
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = TrafficLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# ۵. آموزش مدل (Training Loop)
# ---------------------------------------------------------
print("\nدر حال آموزش شبکه عصبی LSTM با PyTorch...")
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.6f}")

# ---------------------------------------------------------
# ۶. پیش‌بینی و برگرداندن داده‌ها به مقیاس اصلی
# ---------------------------------------------------------
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).numpy()

predictions_actual = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test.numpy())

# ---------------------------------------------------------
# ۷. سیستم کشف آنومالی (Anomaly Detection)
# ---------------------------------------------------------
residuals = np.abs(y_test_actual - predictions_actual)
threshold = np.mean(residuals) + 3 * np.std(residuals)
anomalies = residuals > threshold

print(f"\nحد آستانه تشخیص آنومالی: {threshold:.2f} Mbps")
print(f"تعداد آنومالی‌های/حملات کشف شده در داده تست: {np.sum(anomalies)} مورد")

# ---------------------------------------------------------
# ۸. رسم نمودار
# ---------------------------------------------------------
test_dates = df.index[train_size + LOOK_BACK:]

plt.figure(figsize=(15, 7))
plt.plot(test_dates, y_test_actual, label="ترافیک واقعی شبکه", color="blue", alpha=0.7)
plt.plot(test_dates, predictions_actual, label="پیش‌بینی هوشمند LSTM (PyTorch)", color="green", linestyle="--")

# علامت‌گذاری آنومالی‌ها روی نمودار
plt.scatter(
    test_dates[anomalies.flatten()],
    y_test_actual[anomalies.flatten()],
    color='red', label='آنومالی / حمله کشف‌شده', zorder=5, s=50
)

plt.title("سیستم پیش‌بینی پیشرفته ترافیک شبکه و کشف آنومالی با PyTorch", fontsize=14)
plt.xlabel("زمان")
plt.ylabel("ترافیک مصرفی (Mbps)")
plt.legend()
plt.grid(True)
plt.show()