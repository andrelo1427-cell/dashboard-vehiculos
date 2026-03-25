import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

# ======================
# MODELO REGRESIÓN LOGÍSTICA (con escalado)
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

app.layout = html.Div(style={'backgroundColor': '#f4f6f7', 'padding': '20px'}, children=[

    # ======================
    # TÍTULO + INTRO
    # ======================
    html.H1("🚗 Dashboard Análisis de Vehículos", style={'textAlign': 'center'}),

    html.P(
        "Este dashboard analiza la relación entre las características técnicas de los vehículos y su precio, "
        "utilizando modelos estadísticos para evaluar su capacidad predictiva.",
        style={'textAlign': 'center'}
    ),

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

    # ======================
    # MÉTRICAS
    # ======================
    html.H2("📊 Métricas del modelo"),

    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        html.Div([
            html.H4("R²"),
            html.H3(round(r2, 3))
        ], style={'background': 'white', 'padding': '15px', 'borderRadius': '10px'}),

        html.Div([
            html.H4("MSE"),
            html.H3(round(mse, 2))
        ], style={'background': 'white', 'padding': '15px', 'borderRadius': '10px'}),

        html.Div([
            html.H4("Accuracy"),
            html.H3(round(accuracy, 3))
        ], style={'background': 'white', 'padding': '15px', 'borderRadius': '10px'}),
    ]),

    # ======================
    # GRÁFICOS
    # ======================
    html.H2("📈 Análisis exploratorio"),

    dcc.Graph(id='grafico_dispersion'),

    dcc.Graph(
        figure=px.histogram(df, x="PRECIO", title="Distribución de precios")
    ),

    dcc.Graph(
        figure=px.box(df, y="PRECIO", title="Detección de valores atípicos")
    ),

    # ======================
    # VALIDACIÓN DEL MODELO
    # ======================
    html.H2("📉 Validación del modelo"),

    dcc.Graph(
        figure=px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Valor real', 'y': 'Predicción'},
            title="Comparación valores reales vs predichos"
        )
    ),

    # ======================
    # MATRIZ DE CONFUSIÓN
    # ======================
    html.H2("🤖 Clasificación"),

    dcc.Graph(
        figure=px.imshow(
            cm,
            text_auto=True,
            title="Matriz de confusión",
            labels=dict(x="Predicción", y="Real")
        )
    ),

    # ======================
    # CONCLUSIÓN
    # ======================
    html.H2("📌 Conclusiones"),

    html.P(
        f"El modelo de regresión lineal presenta un R² de {round(r2,3)}, lo que indica una capacidad moderada "
        "para explicar la variabilidad del precio. Por otro lado, el modelo de clasificación alcanza una precisión "
        f"de {round(accuracy,3)}, lo que demuestra una alta capacidad para distinguir entre vehículos costosos y económicos. "
        "Esto sugiere que las variables técnicas tienen una influencia significativa en el precio."
    )
])

# ======================
# CALLBACK
# ======================
@app.callback(
    Output('grafico_dispersion', 'figure'),
    Input('filtro_cilindros', 'value')
)
def actualizar_grafico(cilindros):
    if cilindros is None or len(cilindros) == 0:
        dff = df
    else:
        dff = df[df['CILINDROS'].isin(cilindros)]

    fig = px.scatter(
        dff,
        x="CABALLOS DE FUERZA",
        y="PRECIO",
        color="CILINDROS",
        title="Relación entre potencia y precio"
    )

    return fig

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run_server(debug=False)
