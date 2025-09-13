import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from Regression import Normal

# ===== Initialize app =====
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server

# ===== Load Data =====
vehicle_df = pd.read_csv("car_final_le.csv")

# ===== Load New Model =====
model_new = pickle.load(open("Model/a2-car-price-prediction.model", 'rb'))
mileage_max_power_scalar_new = pickle.load(open("Model/a2-mileage-max-power-scalar.model", 'rb'))
year_scalar_new = pickle.load(open("Model/a2-year-scalar.model", 'rb'))
brand_encoder_new = pickle.load(open("Model/car-brand-encoder.pkl", 'rb'))
fuel_encoder_new = pickle.load(open("Model/brand-fuel.model", 'rb'))

brand_cat_new = list(brand_encoder_new.categories_[0])
fuel_cat_new = list(fuel_encoder_new.classes_)

# ===== Load Old Model =====
model_old = pickle.load(open("Model_old/car-prediction.model", 'rb'))
scaler_old = pickle.load(open("Model_old/car-scalar.model", 'rb'))
label_car_old = pickle.load(open("Model_old/brand-label.model", 'rb'))
fuel_car_old = pickle.load(open("Model_old/brand-fuel.model", 'rb'))

brand_cat_old = list(label_car_old.classes_)
fuel_cat_old = list(fuel_car_old.classes_)
num_cols_old = ['max_power', 'mileage']

# ============================================================
# ===== Sidebar =====
# ============================================================
sidebar = dbc.Col(
    [
        html.H2("🚘 CarDash", className="text-center mb-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="bi bi-speedometer2 me-2"), " Prediction"], href="/", id="link-prediction", active="exact"),
                dbc.NavLink([html.I(className="bi bi-bar-chart-line me-2"), " Stats & Graphs"], href="/stats", id="link-stats", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    width=2,
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "backgroundColor": "#1e1e1e",
        "padding": "20px",
        "height": "100%",
    },
)

# ============================================================
# ===== Hero Section =====
# ============================================================
def hero_section(title, subtitle):
    return dbc.Container(
        [
            html.H1(title, className="display-4 text-center mb-2"),
            html.P(subtitle, className="lead text-center text-muted"),
            html.Hr(),
        ],
        fluid=True,
        className="py-3",
    )

# ============================================================
# ===== Prediction Page =====
# ============================================================
prediction_page = dbc.Container([
    hero_section("🚗 Car Price Prediction", "Compare predictions from New vs Old models"),

    dbc.Card(
        dbc.CardBody([
            # Inputs in 3-column rows
            dbc.Row([
                dbc.Col([dbc.Label("Brand"), dcc.Dropdown(id="brand", options=[{'label': b, 'value': b} for b in brand_cat_new], value=brand_cat_new[0], className="mb-3")], width=4),
                dbc.Col([dbc.Label("Year"), dcc.Input(id="year", type="number", value=2017, className="form-control mb-3")], width=4),
                dbc.Col([dbc.Label("Fuel"), dcc.Dropdown(id="fuel", options=[{'label': f, 'value': f} for f in fuel_cat_new], value=fuel_cat_new[0], className="mb-3")], width=4)
            ]),
            dbc.Row([
                dbc.Col([dbc.Label("Mileage"), dcc.Input(id="mileage", type="number", value=19.42, className="form-control mb-3")], width=4),
                dbc.Col([dbc.Label("Max Power"), dcc.Input(id="max_power", type="number", value=82.4, className="form-control mb-3")], width=4),
                dbc.Col([], width=4)
            ]),
            dbc.Row([
                dbc.Col([dbc.Button("Predict", id="submit", color="primary", className="w-100")], width=4)
            ])
        ]),
        className="shadow-lg mb-4"
    ),

    # Results cards
    dbc.Row([
        dbc.Col([dbc.Card(dbc.CardBody([html.H5("New Model Prediction", className="card-title"), html.Div(id="prediction_result_new", className="fs-4 fw-bold text-success")]), className="shadow-sm")], width=6),
        dbc.Col([dbc.Card(dbc.CardBody([html.H5("Old Model Prediction", className="card-title"), html.Div(id="prediction_result_old", className="fs-4 fw-bold text-info")]), className="shadow-sm")], width=6)
    ])
], fluid=True)

# ============================================================
# ===== Stats & Graphs Page =====
# ============================================================
stats_page = dbc.Container([
    hero_section("📊 Statistics & Graphs", "Explore dataset insights"),

    dbc.Row([
        dbc.Col([dbc.Card(dbc.CardBody([html.H5("Year Distribution"), dcc.Graph(figure=px.histogram(vehicle_df, x="year", nbins=20, title="Car Year Distribution"))]), className="shadow-sm mb-3")], width=6),
        dbc.Col([dbc.Card(dbc.CardBody([html.H5("Fuel Type Count"), dcc.Graph(figure=px.pie(vehicle_df, names="fuel", title="Fuel Type Share"))]), className="shadow-sm mb-3")], width=6),
    ]),

    dbc.Row([
        dbc.Col([dbc.Card(dbc.CardBody([html.H5("Price vs Mileage"), dcc.Graph(figure=px.scatter(vehicle_df, x="mileage", y="selling_price", color="fuel", title="Price vs Mileage"))]), className="shadow-sm mb-3")], width=12)
    ])
], fluid=True)

# ============================================================
# ===== Main Layout =====
# ============================================================
app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        dbc.Col(id="page-content", width={"size": 10, "offset": 2})
    ])
], fluid=True)

# ============================================================
# ===== Routing =====
# ============================================================
@app.callback(
    Output("page-content", "children"),
    [Input("link-prediction", "n_clicks"), Input("link-stats", "n_clicks")]
)
def render_page(pred_click, stats_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        return prediction_page
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "link-stats":
        return stats_page
    return prediction_page

# ============================================================
# ===== Prediction Callback =====
# ============================================================
@app.callback(
    [Output("prediction_result_new", "children"), Output("prediction_result_old", "children")],
    Input("submit", "n_clicks"),
    State("year", "value"),
    State("max_power", "value"),
    State("brand", "value"),
    State("mileage", "value"),
    State("fuel", "value"),
    prevent_initial_call=True
)
def predict_both_models(n, year, max_power, brand, mileage, fuel):
    # New model
    X_new = pd.DataFrame({'year':[year], 'max_power':[max_power], 'mileage':[mileage], 'brand':[brand], 'fuel':[fuel]})
    X_new[['year']] = year_scalar_new.transform(X_new[['year']])
    X_new[['max_power','mileage']] = mileage_max_power_scalar_new.transform(X_new[['max_power','mileage']])
    brand_enc = brand_encoder_new.transform(X_new[['brand']])
    brand_cols = brand_encoder_new.get_feature_names_out(['brand'])
    X_brand = pd.DataFrame(brand_enc, columns=brand_cols, index=X_new.index)
    X_new = pd.concat([X_new.drop(columns=['brand']), X_brand], axis=1)
    X_new['fuel'] = fuel_encoder_new.transform(X_new['fuel'])
    X_new.insert(0,'intercept',1)
    new_price = np.round(np.exp(model_new.predict(X_new)),2)[0]

    # Old model
    X_old = pd.DataFrame({'year':[year], 'max_power':[max_power], 'mileage':[mileage], 'brand':[brand], 'fuel':[fuel]})
    X_old[num_cols_old] = scaler_old.transform(X_old[num_cols_old])
    X_old['brand'] = label_car_old.transform(X_old['brand'])
    X_old['fuel'] = fuel_car_old.transform(X_old['fuel'])
    X_old = X_old[model_old.feature_names_in_]
    old_price = np.round(np.exp(model_old.predict(X_old)),2)[0]

    return f"💰 New Model Price: Rs.{new_price}", f"💰 Old Model Price: Rs.{old_price}"

# ===== Run App =====
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
