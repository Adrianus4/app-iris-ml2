import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2

# ==============================
# CONFIGURACIÓN DB
# ==============================
USER = "postgres.yeoatwmlxgfygafwxwwm"
PASSWORD = "ADR123adr123"
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# ==============================
# CONFIGURACIÓN STREAMLIT
# ==============================
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
st.title("🌸 Predictor de Especies de Iris")

# ==============================
# CONEXIÓN A LA BD
# ==============================
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

try:
    connection = get_connection()
    cursor = connection.cursor()
except Exception as e:
    st.error(f"Error de conexión: {e}")
    st.stop()

# ==============================
# CARGA DE MODELOS
# ==============================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo.")
        return None, None, None

model, scaler, model_info = load_models()

# ==============================
# INPUTS
# ==============================
if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", 0.0, 10.0, 3.0, 0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 4.0, 0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", 0.0, 10.0, 1.0, 0.1)

    # ==============================
    # PREDICCIÓN + INSERT
    # ==============================
    if st.button("Predecir Especie"):
        try:
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]

            target_names = model_info['target_names']
            predicted_species = target_names[prediction]

            # Mostrar resultado
            st.success(f"Especie predicha: **{predicted_species}**")
            st.write(f"Confianza: **{max(probabilities):.1%}**")

            st.write("Probabilidades:")
            for species, prob in zip(target_names, probabilities):
                st.write(f"- {species}: {prob:.1%}")

            # INSERTAR EN LA BD
            cursor.execute("""
                INSERT INTO ml.tb_iris (l_p, l_s, a_s, a_o, prediccion)
                VALUES (%s, %s, %s, %s, %s)
            """, (petal_length, sepal_length, sepal_width, petal_width, predicted_species))

            connection.commit()

            st.success("Datos guardados en la base de datos ✅")

        except Exception as e:
            st.error(f"Error en predicción o inserción: {e}")

# ==============================
# HISTÓRICO
# ==============================
st.header("📊 Histórico de predicciones")

try:
    cursor.execute("""
        SELECT created_at, l_p, l_s, a_s, a_o, prediccion
        FROM ml.tb_iris
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()

    if rows:
        # Convertir a formato tabla
        import pandas as pd
        df = pd.DataFrame(rows, columns=[
            "Fecha", "Largo Pétalo", "Largo Sépalo",
            "Ancho Sépalo", "Ancho Pétalo", "Predicción"
        ])
        st.dataframe(df)
    else:
        st.write("No hay registros aún.")

except Exception as e:
    st.error(f"Error al consultar histórico: {e}")

# ==============================
# CIERRE (OPCIONAL)
# ==============================
# No cerramos conexión porque Streamlit reutiliza recursos cacheados
