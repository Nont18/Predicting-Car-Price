import pickle
import dash
from dash import Dash, html, dcc, State, callback
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc

# Load the trained model
loaded_model = pickle.load(open('model/car_price_prediction.model', 'rb'))

scaler = pickle.load(open('scale/scaler.prep', 'rb'))

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            dbc.Label("max_power"),
            dbc.Input(id="max_power", type="float", placeholder="Put a value for max_power"),
            dbc.Label("mileage"),
            dbc.Input(id="mileage", type="float", placeholder="Put a value for mileage"),
            dbc.Label("km_driven"),
            dbc.Input(id="km_driven", type="float", placeholder="Put a value for km_driven"),
            dbc.Button(id="submit", children="Predict car price", color="primary", className="me-1"),
            dbc.Label("predicted price : "),
            html.Output(id="predicted_price", children="")
        ],
        className="mb-3")
    ])

], fluid=True)

@callback(
    Output(component_id="predicted_price", component_property="children"),
    State(component_id="max_power", component_property="value"),
    State(component_id="mileage", component_property="value"),
    State(component_id="km_driven", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def predict(n_clicks, max_power, mileage, km_driven):
    sample_df = [max_power, mileage, km_driven]
    sample_df = scaler.transform([sample_df])
    return np.exp(loaded_model.predict(sample_df))

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)