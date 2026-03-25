import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# ======================
# CARGAR DATOS
# ======================
df = pd.read_excel("BASE.xlsx")

# ======================
# LIMPIEZA BÁSICA
# ======================
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

y_pred = modelo_lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# ======================
# MODELO REGRESIÓN LOGÍSTICA
# ======================
X2 = df[["CABALLOS DE FUERZA", "CILINDROS"]]
y2 = df["AUTO_CARO"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)

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

    html.H1("🚗 Dashboard Análisis de Vehículos", style={'textAlign': 'center'}),

    html.P("Este dashboard analiza cómo las características del motor influyen en el precio de los vehículos.",
           style={'textAlign': 'center'}),

    # ======================
    # FILTRO
    # ======================
    html.H3("Filtro por número de cilindros"),
    dcc.Dropdown(
        id='filtro_cilindros',
        options=[{'label': str(i), 'value': i} for i in sorted(df['CILINDROS'].unique())],
        value=df['CILINDROS'].unique()[0]
    ),

    # ======================
    # GRÁFICOS
    # ======================
    dcc.Graph(id='grafico_dispersion'),

    dcc.Graph(
        figure=px.histogram(df, x="PRECIO", title="Distribución de precios")
    ),

    dcc.Graph(
        figure=px.box(df, y="PRECIO", title="Valores atípicos (Outliers)")
    ),

    # ======================
    # RESULTADOS MODELOS
    # ======================
    html.H2("📊 Resultados Regresión Lineal"),
    html.P(f"R²: {round(r2, 3)}"),
    html.P(f"MSE: {round(mse, 2)}"),

    dcc.Graph(
        figure=px.scatter(x=y_test, y=y_pred,
                          labels={'x': 'Real', 'y': 'Predicho'},
                          title="Valores reales vs predichos")
    ),

    html.H2("🤖 Regresión Logística"),
    html.P(f"Accuracy: {round(accuracy, 3)}"),

    dcc.Graph(
        figure=px.imshow(cm,
                         text_auto=True,
                         title="Matriz de confusión",
                         labels=dict(x="Predicción", y="Real"))
    ),

    # ======================
    # CONCLUSIÓN
    # ======================
    html.H2("📌 Conclusión"),
    html.P("Los resultados muestran que las características del motor influyen significativamente en el precio de los vehículos. "
           "El modelo de regresión logística alcanza alta precisión, lo que confirma la existencia de patrones claros en los datos.")
])

# ======================
# CALLBACK INTERACTIVO
# ======================
@app.callback(
    Output('grafico_dispersion', 'figure'),
    Input('filtro_cilindros', 'value')
)
def actualizar_grafico(cilindro):
    dff = df[df['CILINDROS'] == cilindro]
    fig = px.scatter(dff,
                     x="CABALLOS DE FUERZA",
                     y="PRECIO",
                     title=f"Potencia vs Precio (Cilindros = {cilindro})")
    return fig


# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)