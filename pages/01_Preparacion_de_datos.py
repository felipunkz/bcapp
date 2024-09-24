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

# Título de la aplicación 
st.title("Predicción de Medicamentos Basada en Condición Médica y Edad") 
 
st.write("""
        A continuación se presenta una vista previa de los datos utilizados para el entrenamiento del modelo:
         """)

df = pd.read_csv("df0.csv")
st.write(df.sample(100)) 

st.write(""" 
        Tras el análisis de los datos de la muestra elegida nos percatamos un objetivo que podíamos conseguir es la **predicción de cual sería el mejor medicamento** considerancdo las caracterísitcas de _'Edad'_ y _'Padecimiento'_ de cada paciente.
         
         Por ello, se eligirán solo esas 3 columnas de la muestra
        """)

code = '''df = df[['Age', 'Medication', 'Medical Condition']]'''
st.code(code, language="python")

df = df[['Age', 'Medication', 'Medical Condition']] 

st.write(df.sample(100)) 

st.write(""" 
         Tras hacer algunas corridas de prueba con los datos originales de la muestra nos percatamos que la naturaleza sintética de los mismos no permitia una precisión adecuada para nuestro objetivo, es por ello que nos vimos en la necesidad de ajustar la cantidad de enfermedades de la muestra a las 10 más frecuentes
        """)

code = '''top_medications = df['Medication'].value_counts().index[:10]'''
st.code(code, language="python")

top_medications = df['Medication'].value_counts().index[:10] 
st.dataframe(top_medications)

st.write(""" 
         A continuación, corresponde ajustar el conjunto de datos para que éste incluya únicamente los datos relacionados con los medicamentos elegidos:
        """)

code = '''
        df_reduced = df[df['Medication'].isin(top_medications)] 
        
        X = df_reduced[['Age', 'Medical Condition']] 
        y = df_reduced['Medication'] 
        '''
st.code(code, language="python")

# Filtrar el DataFrame para incluir solo esos medicamentos 
df_reduced = df[df['Medication'].isin(top_medications)] 

# Preparar los datos 
X = df_reduced[['Age', 'Medical Condition']] 
y = df_reduced['Medication'] 

col1, buf, col2 = st.columns([3, 1, 3])

col1.subheader("Atributos de la muestra")
col1.write(X.head(15))

col2.subheader("Variable objetivo de la muestra")
col2.write(y.head(15))

if 'clean_df' not in st.session_state:
    st.session_state['clean_df'] = df_reduced
    st.session_state['reduced_attr'] = X
    st.session_state['reduced_tgt'] = y