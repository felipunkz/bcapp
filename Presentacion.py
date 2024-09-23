import streamlit as st

st.set_page_config(
    page_title="Presentación del proyecto",
    page_icon="👨‍⚕️",
)

st.write("# Proyecto final para Bootcamp de ciencia de datos - Código Facilito")

st.markdown(
    """
    ## Proyecto:

    Entrenamiento de modelo de ML para determinar cual es el medicamento más adecuado a recetar dadas algunas condiciones de edad y tipo de enfermedad

    ## Objetivo:

    Se ha tomado un dataset de prueba que puede referenciarse en la siguiente URL [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset/data)

"""
)