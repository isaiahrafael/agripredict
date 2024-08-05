from flask import Flask, request, render_template, redirect, url_for, send_file, Response
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import os
import time

# Set Matplotlib backend to 'Agg' for non-GUI rendering
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/isaiah/Documents/flask_forecast/uploads'
app.config['STATIC_FOLDER'] = '/Users/isaiah/Documents/flask_forecast/static'

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

progress = 0

def process_auto_arima_forecast(file_path, forecast_steps=24, years=10):
    global progress
    data = pd.read_csv(file_path, index_col='Date')

    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index, errors='coerce')

    data.ffill(inplace=True)
    monthly_data = data.resample('M').mean()
    end_date = monthly_data.index[-1]
    start_date = end_date - pd.DateOffset(years=10)
    filtered_data = monthly_data.loc[start_date:end_date]
    commodity_columns = filtered_data.columns

    forecast_results = []

    for column in commodity_columns:
        try:
            model = pm.auto_arima(filtered_data[column], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
            forecast = model.predict(n_periods=forecast_steps, return_conf_int=True)
            forecast_mean, forecast_ci = forecast
            forecast_dates = pd.date_range(start=filtered_data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                f'{column} Forecasted Monthly Average': forecast_mean
            })

            historical_df = filtered_data[[column]].reset_index()
            historical_path = os.path.join(app.config['UPLOAD_FOLDER'], f'historical_{column.lower().replace(" ", "_")}.csv')
            forecast_path = os.path.join(app.config['UPLOAD_FOLDER'], f'forecasted_{column.lower().replace(" ", "_")}.csv')
            combined_path = os.path.join(app.config['UPLOAD_FOLDER'], f'combined_{column.lower().replace(" ", "_")}.csv')

            historical_df.to_csv(historical_path, index=False)
            forecast_df.to_csv(forecast_path, index=False)
            combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
            combined_df.to_csv(combined_path, index=False)

            combined_forecast_mean = pd.concat([pd.Series(filtered_data[column].iloc[-1]), pd.Series(forecast_mean)], ignore_index=True)
            combined_forecast_ci_lower = pd.concat([pd.Series(filtered_data[column].iloc[-1]), pd.Series(forecast_ci[:, 0])], ignore_index=True)
            combined_forecast_ci_upper = pd.concat([pd.Series(filtered_data[column].iloc[-1]), pd.Series(forecast_ci[:, 1])], ignore_index=True)
            combined_forecast_dates = pd.date_range(start=filtered_data.index[-1], periods=forecast_steps + 1, freq='M')

            plt.figure(figsize=(14, 7))
            plt.plot(filtered_data[column], label=f'Historical {column} Prices (Last {years} Years)', color='blue')
            plt.plot(combined_forecast_dates, combined_forecast_mean, label=f'Forecasted {column} Prices (Next 2 Years)', color='orange')
            plt.fill_between(combined_forecast_dates, combined_forecast_ci_lower, combined_forecast_ci_upper, color='orange', alpha=0.2)
            plt.title(f'{column} Prices Forecast (Next 2 Years)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plot_filename = f'{column.lower().replace(" ", "_")}_forecast.png'
            plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
            plt.savefig(plot_path)
            plt.close()

            forecast_results.append({
                'column': column,
                'historical_path': historical_path,
                'forecast_path': forecast_path,
                'combined_path': combined_path,
                'plot_path': plot_filename  # Save just the filename, not the full path
            })
        except Exception as e:
            print(f"Could not fit auto_arima model for {column}: {e}")

        progress += int(100 / len(commodity_columns))

    return forecast_results

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
        forecast_results = process_auto_arima_forecast(file_path)
        return render_template('results.html', results=forecast_results)

@app.route('/progress')
def progress():
    def generate():
        global progress
        while progress < 100:
            yield f"data:{progress}\n\n"
            time.sleep(1)
        yield f"data:100\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
