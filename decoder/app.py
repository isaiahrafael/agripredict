from flask import Flask, request, render_template, redirect, url_for, send_file, Response
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
from prophet import Prophet
import os
import time

# Set Matplotlib backend to 'Agg' for non-GUI rendering
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/thisi/decoder/uploads'
app.config['STATIC_FOLDER'] = '/Users/thisi/decoder/static'

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

progress = 0

def calculate_mse(actual, predicted):
    return mean_squared_error(actual, predicted)

def process_sarima(data, forecast_steps=24):
    forecast_results = []
    progress_increment = 100 / len(data.columns) if not data.empty else 0
    mse_scores = {}

    for column in data.columns:
        try:
            model = pm.auto_arima(data[column], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
            forecast = model.predict(n_periods=forecast_steps)
            actual = data[column].iloc[-forecast_steps:]
            mse = calculate_mse(actual, forecast)
            mse_scores[column] = mse

            forecast_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

            plt.figure(figsize=(14, 7))
            plt.plot(data[column], label=f'Historical {column} Prices')
            plt.plot(forecast_dates, forecast, label=f'Forecasted {column} Prices', color='orange')
            plt.title(f'{column} Prices Forecast - SARIMA')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plot_filename = f'sarima_{column.lower().replace(" ", "_")}_forecast.png'
            plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
            plt.savefig(plot_path)
            plt.close()

            forecast_results.append({
                'column': column,
                'mse': mse,
                'plot_path': plot_filename,
                'model': 'SARIMA'
            })
        except Exception as e:
            print(f"Could not fit SARIMA model for {column}: {e}")

        global progress
        progress += progress_increment

    return forecast_results, mse_scores

def process_prophet(data, forecast_steps=24):
    forecast_results = []
    progress_increment = 100 / len(data.columns) if not data.empty else 0
    mse_scores = {}

    for column in data.columns:
        try:
            df = data[[column]].reset_index().rename(columns={'Date': 'ds', column: 'y'})
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=forecast_steps, freq='M')
            forecast = model.predict(future)

            actual = df['y'].iloc[-forecast_steps:]
            predicted = forecast['yhat'].iloc[-forecast_steps:]
            mse = calculate_mse(actual, predicted)
            mse_scores[column] = mse

            plt.figure(figsize=(14, 7))
            plt.plot(df['ds'], df['y'], label=f'Historical {column} Prices')
            plt.plot(forecast['ds'], forecast['yhat'], label=f'Forecasted {column} Prices', color='green')
            plt.title(f'{column} Prices Forecast - Prophet')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plot_filename = f'prophet_{column.lower().replace(" ", "_")}_forecast.png'
            plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
            plt.savefig(plot_path)
            plt.close()

            forecast_results.append({
                'column': column,
                'mse': mse,
                'plot_path': plot_filename,
                'model': 'Prophet'
            })
        except Exception as e:
            print(f"Could not fit Prophet model for {column}: {e}")

        global progress
        progress += progress_increment

    return forecast_results, mse_scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    progress = 0
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        data = pd.read_csv(file_path, index_col='Date')
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index, errors='coerce')
        data.ffill(inplace=True)
        monthly_data = data.resample('M').mean()
        end_date = monthly_data.index[-1]
        start_date = end_date - pd.DateOffset(years=10)
        filtered_data = monthly_data.loc[start_date:end_date]

        sarima_results, sarima_mse_scores = process_sarima(filtered_data)
        prophet_results, prophet_mse_scores = process_prophet(filtered_data)

        comparison_results = []
        for column in filtered_data.columns:
            sarima_mse = sarima_mse_scores.get(column, float('inf'))
            prophet_mse = prophet_mse_scores.get(column, float('inf'))
            comparison_results.append({
                'column': column,
                'sarima_mse': sarima_mse,
                'prophet_mse': prophet_mse,
                'better_model': 'SARIMA' if sarima_mse < prophet_mse else 'Prophet',
                'sarima_plot_path': f'sarima_{column.lower().replace(" ", "_")}_forecast.png',
                'prophet_plot_path': f'prophet_{column.lower().replace(" ", "_")}_forecast.png'
            })

        return render_template('results.html', results=comparison_results)

@app.route('/progress')
def progress():
    def generate():
        global progress
        while progress < 100:
            yield f"data:{int(progress)}\n\n"
            time.sleep(1)
        yield f"data:100\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
