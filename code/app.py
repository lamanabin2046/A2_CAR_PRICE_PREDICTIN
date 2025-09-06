# app.py
import dash
from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px

# Dummy dataset (replace with real CSV if you have it)
df = pd.DataFrame({
    "year": np.random.randint(2010, 2021, 100),
    "mileage": np.random.uniform(10, 25, 100),
    "max_power": np.random.uniform(50, 120, 100),
    "selling_price": np.random.uniform(2000, 15000, 100),
    "brand": np.random.choice(["Maruti", "Hyundai", "Toyota"], 100),
    "fuel": np.random.choice(["Petrol", "Diesel", "CNG"], 100),
})

# Initialize app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ===== Navbar =====
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("🚗 Car Price Dashboard", className="ms-2 fw-bold"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Prediction", href="#prediction")),
            dbc.NavItem(dbc.NavLink("Analytics", href="#analytics")),
        ], className="ms-auto", navbar=True)
    ]),
    color="dark",
    dark=True,
    sticky="top",
)

# ===== Hero Section =====
hero = dbc.Container([
dbc.Row([
    dbc.Col([
        html.H1(
            "Welcome to the Car Price Prediction App 🚘",
            className="display-5 fw-bold"
        ),
        html.P(
            "Predict car prices, analyze trends, and visualize data with ease. "
            "Use the sidebar to navigate between tools.",
            className="lead"
        ),
    ], width=12, className="text-center")  # 👈 center the text inside column
], className="my-4 justify-content-center"),
], fluid=True)

# ===== Sidebar =====
sidebar = dbc.Col([
    html.H5("Navigation", className="text-center fw-bold"),
    html.Hr(),
    dbc.Nav(
        [
            dbc.NavLink("🏠 Home", href="#", active="exact"),
            dbc.NavLink("📊 Prediction", href="#prediction", active="exact"),
            dbc.NavLink("📈 Analytics", href="#analytics", active="exact"),
        ],
        vertical=True,
        pills=True,
    ),
], width=2, className="bg-light vh-100 p-3")

# ===== Content (Tabs inside main area) =====
content = dbc.Col([
    dcc.Tabs([

        # Prediction Tab
        dcc.Tab(label="Prediction", id="prediction", children=[
            dbc.Card(dbc.CardBody([

                dbc.Row([
                    dbc.Col([dbc.Label("Brand"),
                             dcc.Dropdown(["Maruti", "Hyundai", "Toyota"], "Maruti", id="brand")], width=4),

                    dbc.Col([dbc.Label("Year"),
                             dcc.Dropdown(list(range(2010, 2021)), 2017, id="year")], width=4),

                    dbc.Col([dbc.Label("Fuel"),
                             dcc.Dropdown(["Petrol", "Diesel", "CNG"], "Diesel", id="fuel")], width=4),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([dbc.Label("Mileage (km/l)"),
                             dcc.Input(id="mileage", type="number", value=18, style={"width": "100%"})], width=6),

                    dbc.Col([dbc.Label("Max Power (bhp)"),
                             dcc.Input(id="max_power", type="number", value=85, style={"width": "100%"})], width=6),
                ], className="mb-3"),

                dbc.Button("Predict Price", id="submit", color="primary", className="w-100 mb-3"),
                html.Div(id="prediction_result", className="text-center fs-4 fw-bold")
            ]))
        ]),

        # Analytics Tab
        dcc.Tab(label="Analytics", id="analytics", children=[
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Choose X-axis"),
                        dcc.Dropdown(["year", "mileage", "max_power"], "year", id="x_axis")
                    ], width=4),

                    dbc.Col([
                        dbc.Label("Choose Y-axis"),
                        dcc.Dropdown(["selling_price"], "selling_price", id="y_axis")
                    ], width=4),
                ], className="mb-3"),

                dcc.Graph(id="scatter_plot"),

                html.Hr(),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Avg Price"),
                            html.H4(f"${df['selling_price'].mean():,.2f}")
                        ])
                    ], color="primary", inverse=True), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Median Mileage"),
                            html.H4(f"{df['mileage'].median():.1f} km/l")
                        ])
                    ], color="success", inverse=True), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Avg Max Power"),
                            html.H4(f"{df['max_power'].mean():.1f} bhp")
                        ])
                    ], color="info", inverse=True), width=3),
                ], className="mt-3"),
            ]))
        ]),
    ])
], width=10)

# ===== Footer =====
footer = dbc.Container([
    html.Hr(),
    html.P("© 2025 Car Price Dashboard | Built with Dash + Bootstrap", className="text-center text-muted"),
], fluid=True)

# ===== App Layout =====
app.layout = html.Div([
    navbar,
    hero,
    dbc.Container([
        dbc.Row([
            sidebar,
            content
        ])
    ], fluid=True),
    footer
])

# ===== Callbacks =====
@callback(
    Output("prediction_result", "children"),
    Input("submit", "n_clicks"),
    State("year", "value"),
    State("max_power", "value"),
    State("brand", "value"),
    State("mileage", "value"),
    State("fuel", "value"),
    prevent_initial_call=True
)
def predict_price(n, year, max_power, brand, mileage, fuel):
    # Dummy price formula
    price = 5000 + (year - 2010) * 800 + max_power * 40 + mileage * 30
    return f"💰 Predicted Price: ${price:,.2f}"

@callback(
    Output("scatter_plot", "figure"),
    Input("x_axis", "value"),
    Input("y_axis", "value")
)
def update_graph(x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, color="fuel", hover_data=["brand"])
    return fig

# Run
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
