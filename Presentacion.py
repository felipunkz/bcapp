import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder 
from sklearn.ensemble import RandomForestClassifier, StackingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Input 
from tensorflow.keras.utils import to_categorical 
from sklearn.base import BaseEstimator, ClassifierMixin 


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

    ## El equipo:

    Nuestro equipo se compone de _Alexyz Maynez_ (Médico) y _Felipe Nieves_ (Infraestructura de TI). Ambos con una deseo incansable de mejorar y aprender. Buscamos esta formación para incrementar nuestros conocimientos y ¿Por qué no? Considerar la posibilidad de dar un giro profesional a nuestra vida.

    Es importante mencionar que la experiencia en médicina de Alexyz ha jugado un rol importantísimo en la determinación de la utilidad y mejora de los datos elegidos para nuestro proyecto.

"""
)

st.header(":red[AVISO:]", divider="red")
st.write(""" 
         _Desafortunadamente, el dataset seleccionado resultó ser creado con datos sintéticos por lo que la distribución entre los datos y las relaciones entre los atributos distaron mucho representar datos de la vida real, como puede notarse en este comentario: [Unrealistic distribution of values](https://www.kaggle.com/datasets/prasad22/healthcare-dataset/discussion/467020)._

         _Sabemos que esta es razón suficiente para descartar el proyecto y elegir otro conjunto de datos, lo intentamos pero al ser los datos médicos de ídole tan privada fue difícil encontrar datos apropiados, por lo que nos aventuramos con este y tratamos de mostrar utilizarlo para mostrar lo aprendido dentro del curso._

        """)