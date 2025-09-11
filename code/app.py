# app.py
import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from Regression import Normal  # your regression class

# ===== Initialize app =====
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server

# ===== Load data =====
vehicle_df = pd.read_csv("Cars.csv")           # Original dataset
vehicle_df_graph = pd.read_csv("car_final_le.csv")  # For plotting

# ===== Load models and preprocessors =====
model = pickle.load(open("Model/a2-car-price-prediction.model", 'rb'))
mileage_max_power_scalar = pickle.load(open("Model/a2-mileage-max-power-scalar.model", 'rb'))
year_scalar = pickle.load(open("Model/a2-year-scalar.model", 'rb'))
brand_encoder = pickle.load(open("Model/car-brand-encoder.pkl", 'rb'))  # OneHotEncoder
fuel_encoder = pickle.load(open("Model/brand-fuel.model", 'rb'))       # LabelEncoder

brand_cat = list(brand_encoder.categories_[0])
fuel_cat = list(fuel_encoder.classes_)

default_values = {
    'year': 2017,
    'max_power': 82.4,
    'mileage': 19.42,
    'brand': 'Maruti',
    'fuel': 'Diesel'
}

# ===== Sidebar =====
sidebar = dbc.Col(
    [
        html.H3("DASH 🚗", className="text-center"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", id="page-home", active="exact"),
                dbc.NavLink("Prediction", href="/prediction", id="page-prediction", active="exact"),
                dbc.NavLink("Graph", href="/graph", id="page-graph", active="exact"),
                dbc.NavLink("Contact", href="/contact", id="page-contact", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    width=2,
    style={"position": "fixed", "height": "100%", "background-color": "#222", "padding": "20px"},
)

home_page = html.Div([
    # Hero Section
    html.Div([
        html.H3("Welcome to Our Car Company", className="display-3"),
        html.P("Predict car prices accurately and explore feature relationships!", className="lead"),
    ], className="hero"),

    # Car section
    html.Div([
        html.Img(src="https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg",
                 className="car-image"),
        html.Img(src="https://images.pexels.com/photos/116675/pexels-photo-116675.jpeg",
                 className="car-image"),
        html.Img(src="https://images.pexels.com/photos/4639907/pexels-photo-4639907.jpeg",
                 className="car-image"),
    ], className="car-container")
])


# ===== Prediction Page =====
# ===== Prediction Page =====
prediction_page = html.Div([

    # Top Hero Section
    html.Div([
        html.H2("🚗 Car Price Prediction", className="text-center display-4"),
        html.P("Enter car details below to predict the selling price.", className="text-center lead"),
    ], className="prediction-hero", style={"padding": "40px 0", "background-color": "#60bf6c", "color": "white"}),

    # Full Page Form
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Row 1: Brand, Year, Fuel
                        dbc.Row([
                            dbc.Col([
                                html.Label("Brand", className="form-label"),
                                dcc.Dropdown(
                                    id="brand",
                                    options=[{'label': b, 'value': b} for b in brand_cat],
                                    value=default_values['brand'],
                                    style={"width": "100%", "margin-bottom": "15px"}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Year", className="form-label"),
                                dcc.Input(
                                    id="year", type="number",
                                    value=default_values['year'],
                                    style={"width": "100%", "margin-bottom": "15px", "padding": "10px"}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Fuel", className="form-label"),
                                dcc.Dropdown(
                                    id="fuel",
                                    options=[{'label': f, 'value': f} for f in fuel_cat],
                                    value=default_values['fuel'],
                                    style={"width": "100%", "margin-bottom": "15px"}
                                )
                            ], width=4),
                        ], className="mb-3"),

                        # Row 2: Mileage, Max Power
                        dbc.Row([
                            dbc.Col([
                                html.Label("Mileage", className="form-label"),
                                dcc.Input(
                                    id="mileage", type="number",
                                    value=default_values['mileage'],
                                    style={"width": "100%", "margin-bottom": "15px", "padding": "10px"}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Max Power", className="form-label"),
                                dcc.Input(
                                    id="max_power", type="number",
                                    value=default_values['max_power'],
                                    style={"width": "100%", "margin-bottom": "15px", "padding": "10px"}
                                )
                            ], width=6),
                        ], className="mb-3"),

                        # Submit button
                        dbc.Button("Predict Price", id="submit", color="success", className="w-100 mb-3"),

                        # Prediction result
                        html.Div(id="prediction_result", className="text-center fs-4 fw-bold")
                    ])
                ], style={"min-height": "70vh"})  # card takes most of page height
            ], width=12)
        ], style={"margin-top": "20px"})
    ], fluid=True)
])



# ===== Graph Page =====
graph_page = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("📊 Explore Selling Prices vs Features", className="text-center mb-3"),
                        dbc.Label("Select Feature for X-axis:"),
                        dcc.Dropdown(
                            id="feature_x",
                            options=[{'label': col, 'value': col} for col in ['year', 'mileage', 'max_power']],
                            value='year',
                            style={'width': '50%', 'margin-bottom': '20px'}
                        ),
                        dcc.Graph(id="price_vs_feature_graph", style={'height': '70vh'})
                    ])
                ], style={"min-height": "80vh", "background-color": "rgba(0,0,0,0.7)", "border-radius": "15px", "padding": "20px"})
            ], width=12)  # Full width column
        ])
    ], fluid=True)
])

# ===== Contact Page =====
contact_page = html.Div([
    dbc.Container([
        html.H2("Contact Us", className="text-center"),
        html.P("Email: info@ourcars.com | Phone: +66936652501", className="text-center")
    ])
])

# ===== App Layout =====
app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        dbc.Col(id="page-content", width={"size":10, "offset":2})
    ])
], fluid=True)

# ===== Page Router Callback =====
@app.callback(
    Output("page-content", "children"),
    [Input("page-home", "n_clicks"),
     Input("page-prediction", "n_clicks"),
     Input("page-graph", "n_clicks"),
     Input("page-contact", "n_clicks")]
)
def render_page(home, pred, graph, contact):
    ctx = dash.callback_context
    if not ctx.triggered:
        return home_page
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "page-home":
        return home_page
    elif button_id == "page-prediction":
        return prediction_page
    elif button_id == "page-graph":
        return graph_page
    elif button_id == "page-contact":
        return contact_page
    return home_page

# ===== Prediction Callback =====
@app.callback(
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
    features = {
        'year': year or default_values['year'],
        'max_power': max_power or default_values['max_power'],
        'mileage': mileage or default_values['mileage'],
        'brand': brand or default_values['brand'],
        'fuel': fuel or default_values['fuel']
    }

    X = pd.DataFrame(features, index=[0])
    X[['year']] = year_scalar.transform(X[['year']])
    X[['max_power', 'mileage']] = mileage_max_power_scalar.transform(X[['max_power', 'mileage']])
    brand_encoded = brand_encoder.transform(X[['brand']])
    brand_cols = brand_encoder.get_feature_names_out(['brand'])
    X_brand = pd.DataFrame(brand_encoded, columns=brand_cols, index=X.index)
    X = pd.concat([X.drop(columns=['brand']), X_brand], axis=1)
    X['fuel'] = fuel_encoder.transform(X['fuel'])
    X.insert(0, 'intercept', 1)

    price = np.round(np.exp(model.predict(X)), 2)[0]
    return f"💰 Predicted Price: ${price}"

# ===== Graph Callback =====
@app.callback(
    Output("price_vs_feature_graph", "figure"),
    Input("feature_x", "value")
)
def update_graph(feature_x):
    fig = px.scatter(vehicle_df_graph, x=feature_x, y='selling_price',
                     color='fuel', hover_data=['brand', 'mileage', 'max_power'],
                     title=f"Selling Price vs {feature_x}")
   
    fig.update_layout(
        template='plotly_dark',          # optional, keeps dark theme
        plot_bgcolor='rgba(28, 28, 28, 0.8)',  # plot area
        paper_bgcolor='rgba(0, 0, 0, 0)',      # card background is already set
        font=dict(color='white')          # axis labels & titles
    )
    
    return fig

# ===== Home page car sliding callback =====
@app.callback(
    [Output("car1", "className"),
     Output("car2", "className"),
     Output("car3", "className")],
    Input("slide-btn", "n_clicks"),
    prevent_initial_call=True
)
def slide_cars(n):
    return ["car-image car-move", "car-image car-move", "car-image car-move"]

# ===== Run App =====
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
