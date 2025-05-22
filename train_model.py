import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Burada beğeni sayısı rastgele oluşturuluyor, gerçek verilerle değiştir!
data = pd.read_csv("data/features.csv")
data['likes'] = np.random.randint(100, 1000, size=len(data))

X = data[['brightness', 'sharpness', 'color_r', 'color_g', 'color_b']]
y = data['likes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")

joblib.dump(model, 'data/engagement_model.pkl')