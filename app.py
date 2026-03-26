import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# ======================
# CARGAR DATOS
# ======================
df = pd.read_excel("BASE.xlsx")
df = df.dropna()

# ======================
# VARIABLE LOGÍSTICA
# ======================
mediana_precio = df["PRECIO"].median()
df["AUTO_CARO"] = (df["PRECIO"] > mediana_precio).astype(int)

# ======================
# MODELO REGRESIÓN LINEAL
# ======================
X = df[["CABALLOS DE FUERZA", "CILINDROS"]]
y = df["PRECIO"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

y_pred = modelo_lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# VALIDACIÓN CRUZADA
cv_scores = cross_val_score(modelo_lr, X, y, cv=5, scoring='r2')
cv_mean = cv_scores.mean()

# ======================
# MODELO LOGÍSTICO
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, df["AUTO_CARO"], test_size=0.2, random_state=42
)

modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train2, y_train2)

y_pred2 = modelo_log.predict(X_test2)

accuracy = accuracy_score(y_test2, y_pred2)
cm = confusion_matrix(y_test2, y_pred2)

# ======================
# APP DASH
# ======================
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#ecf0f1', 'padding': '20px'}, children=[

    html.H1("🚗 Dashboard Avanzado de Vehículos", style={'textAlign': 'center'}),

    html.P("Análisis estadístico con validación, incertidumbre y modelos predictivos",
           style={'textAlign': 'center'}),

    # ======================
    # FILTROS
    # ======================
    html.H3("🔍 Filtros"),

    dcc.Dropdown(
        id='filtro_cilindros',
        options=[{'label': str(i), 'value': i} for i in sorted(df['CILINDROS'].unique())],
        multi=True,
        placeholder="Selecciona cilindros"
    ),

    dcc.Slider(
        id='filtro_precio',
        min=df["PRECIO"].min(),
        max=df["PRECIO"].max(),
        step=1000,
        value=df["PRECIO"].max(),
        tooltip={"placement": "bottom"},
    ),

    html.Br(),

    # ======================
    # MÉTRICAS
    # ======================
    html.H2("📊 Métricas del modelo"),

    html.Div(style={'display': 'flex', 'gap': '15px'}, children=[

        html.Div([html.H4("R²"), html.H3(round(r2, 3))],
                 style={'background': '#3498db', 'color': 'white', 'padding': '15px', 'borderRadius': '10px'}),

        html.Div([html.H4("RMSE"), html.H3(round(rmse, 2))],
                 style={'background': '#e67e22', 'color': 'white', 'padding': '15px', 'borderRadius': '10px'}),

        html.Div([html.H4("Accuracy"), html.H3(round(accuracy, 3))],
                 style={'background': '#2ecc71', 'color': 'white', 'padding': '15px', 'borderRadius': '10px'}),

        html.Div([html.H4("CV R²"), html.H3(round(cv_mean, 3))],
                 style={'background': '#9b59b6', 'color': 'white', 'padding': '15px', 'borderRadius': '10px'}),
    ]),

    # ======================
    # GRÁFICOS
    # ======================
    html.H2("📈 Análisis"),

    dcc.Graph(id='grafico_dispersion'),

    dcc.Graph(id='histograma'),

    # ======================
    # VALIDACIÓN
    # ======================
    html.H2("📉 Validación"),

    dcc.Graph(
        figure=px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Real', 'y': 'Predicción'},
            title="Valores reales vs predichos"
        )
    ),

    # ======================
    # MATRIZ DE CONFUSIÓN
    # ======================
    html.H2("🤖 Clasificación"),

    dcc.Graph(
        figure=px.imshow(cm, text_auto=True, title="Matriz de confusión")
    ),

    # ======================
    # CONCLUSIÓN
    # ======================
    html.H2("📌 Conclusión"),

    html.P(
        f"El modelo presenta un R² de {round(r2,3)} y un RMSE de {round(rmse,2)}, "
        f"con validación cruzada promedio de {round(cv_mean,3)}. "
        f"La clasificación alcanza un accuracy de {round(accuracy,3)}, "
        "lo que indica una buena capacidad predictiva."
    )
])

# ======================
# CALLBACK
# =================
