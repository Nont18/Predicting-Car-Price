import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dash.dependencies import Input, Output
import pickle
import numpy as np

# Load the trained model and scaler
loaded_model = pickle.load(open('model/car_price_prediction.model', 'rb'))
scaler = pickle.load(open('scale/scaler.prep', 'rb'))

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Welcome to Car Price Prediction System"),
            dbc.CardGroup([
                dbc.Label("Max Power"),
                dcc.Input(id="max_power", type="number", placeholder="Enter max power", debounce=True),
                dbc.Label("Mileage"),
                dcc.Input(id="mileage", type="number", placeholder="Enter mileage", debounce=True),
                dbc.Label("Kilometers Driven"),
                dcc.Input(id="km_driven", type="number", placeholder="Enter kilometers driven", debounce=True),
            ]),
            dbc.Label("Predicted Price:"),
            html.Div(id="predicted_price", className="lead"),
        ], width=10),
    ]),
], fluid=True)

@app.callback(
    Output("predicted_price", "children"),
    Input("max_power", "value"),
    Input("mileage", "value"),
    Input("km_driven", "value"),
)
def predict(max_power, mileage, km_driven):
    if max_power is not None and mileage is not None and km_driven is not None:
        # Scale the input data
        input_data = np.array([[max_power, mileage, km_driven]])
        scaled_input = scaler.transform(input_data)

        # Make a prediction
        predicted_price = np.exp(loaded_model.predict(scaled_input))[0]
        return f"Predicted Price: ${predicted_price:.2f}"
    else:
        return "Enter valid input"

if __name__ == '__main__':
    app.run_server(debug=True)
