import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import joblib

def train_and_save_model():
    # Load data
    df = pd.read_excel('../palm_oil.xlsx')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Create features
    start_date = df['date'].min()
    df['days'] = (df['date'] - start_date).dt.days

    # Split data
    X = df[['days']]
    y = df['nilai']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Prepare historical data for visualization
    historical_data = {
        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'actual': y.tolist(),
        'predicted': y_pred.tolist()
    }

    # Save model and metrics
    joblib.dump({
        'model': model,
        'metrics': {'MAE': mae, 'MAPE': mape, 'RMSE': rmse},
        'start_date': start_date,
        'historical_data': historical_data
    }, 'model.pkl')

if __name__ == '__main__':
    train_and_save_model()