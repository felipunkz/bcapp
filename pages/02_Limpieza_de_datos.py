import streamlit as st
import pandas as pd
import numpy as np

# Título de la aplicación
st.title('Proceso de Limpieza de Datos')

# Paso 1: Cargar los datos
st.header('1. Cargar los datos')
st.write('En este paso, cargamos el archivo CSV que contiene nuestro dataset de pacientes y revisamos si hay valores faltantes.')
code = '''
import pandas as pd

# Carga el archivo CSV usando Pandas
df = pd.read_csv('df0.csv')

# Ver si hay algún valor faltante en el DataFrame
print(df.isnull().sum())
'''
st.code(code, language='python')

# Mostrar tabla de valores nulos
df = pd.read_csv('df0.csv')
missing_values = df.isnull().sum()

# Mostrar la tabla de valores nulos en Streamlit
st.subheader('Tabla de valores nulos')
st.write('La siguiente tabla muestra el número de valores faltantes por columna:')
st.table(missing_values)

# Paso 2: Modificación del "Billing Amount"
st.header('2. Modificación del "Billing Amount"')
st.write('Para hacer los resultados más heterogéneos, modificamos el monto de facturación de un porcentaje de las filas.')
code = '''
# Reducimos la cantidad del "Billing Amount" para hacer los resultados más heterogéneos.

# Definir el porcentaje de filas a modificar y el nuevo valor
porcentaje_a_modificar = 35  # Porcentaje de filas a modificar
nuevo_valor = 10500  # Nuevo valor con el que reemplazar

# Calcular el número de filas a modificar
num_filas = len(df)
num_filas_a_modificar = int(num_filas * porcentaje_a_modificar / 100)

# Seleccionar aleatoriamente las filas a modificar
filas_a_modificar = np.random.choice(df.index, size=num_filas_a_modificar, replace=False)

# Reemplazar los valores en las filas seleccionadas
df.loc[filas_a_modificar, 'Billing Amount'] = nuevo_valor
'''
st.code(code, language='python')

# Paso 3: Transformación de datos
st.header('3. Transformación de Fechas')
st.write('Transformamos las columnas de fechas para asegurarnos de que estén en el formato correcto.')
code = '''
#Transformación de datos
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
'''
st.code(code, language='python')

# Paso 4: Eliminación de columnas no necesarias
st.header('4. Eliminación de columnas no necesarias')
st.write('Se eliminan columnas que no son relevantes para el análisis.')
code = '''
# Elimina las columnas no necesarias
df_dropped = df.drop(columns=['Doctor', 'Hospital', 'Room Number', 'Name', 'Test Results'])
'''
st.code(code, language='python')

# Paso 5: Reducción del número de pacientes para hacer el dataset más diverso
st.header('5. Reducción de pacientes para mayor diversidad')
st.write('Se elimina una cantidad aleatoria de pacientes en base a su género para hacer el dataset más diverso.')
code = '''
porcentajes = {
    'Female': 0.12,
    'Male': 0.26
}

indices_a_eliminar = []

for valor, porcentaje in porcentajes.items():
  filas_objeto = df_dropped[df_dropped['Gender'] == valor]
  num_filas_a_eliminar = int(len(filas_objeto) * porcentaje)
  filas_a_eliminar_aleatorias = filas_objeto.sample(n=num_filas_a_eliminar, random_state=1)
  indices_a_eliminar.extend(filas_a_eliminar_aleatorias.index)

df_modificado = df_dropped.drop(index=indices_a_eliminar)
'''
st.code(code, language='python')

# Paso 6: Reducción del número de aseguradoras
st.header('6. Reducción del número de aseguradoras')
st.write('Reducimos la cantidad de pacientes por cada proveedor de seguro para simular una distribución más realista.')
code = '''
porcentajes_insurance = {
    'Cigna': 0.32,
    'Medicare': 0.26,
    'Aetna' : 0.18,
    'UnitedHealthcare': 0.12,
    'Blue Cross': 0.4
}

indices_a_eliminar = []

for valor, porcentaje in porcentajes_insurance.items():
  filas_objeto1 = df_modificado[df_modificado['Insurance Provider'] == valor]
  num_filas_a_eliminar1 = int(len(filas_objeto1) * porcentaje)
  filas_a_eliminar_aleatorias1 = filas_objeto1.sample(n=num_filas_a_eliminar1, random_state=1)
  indices_a_eliminar.extend(filas_a_eliminar_aleatorias1.index)

df_modificado1 = df_modificado.drop(index=indices_a_eliminar)
'''
st.code(code, language='python')

# Paso 7: Ajuste de las condiciones médicas
st.header('7. Ajuste de las condiciones médicas')
st.write('Se modifica el número de pacientes por condición médica para que el dataset tenga una distribución más realista.')
code = '''
porcentajes_p = {
    'Diabetes':0.16,
    'Obesity':0.2,
    'Cancer':0.7,
    'Hypertension':0.3
}

indices_a_eliminar = []

for valor, porcentaje in porcentajes_p.items():
  filas_objeto2 = df_modificado1[df_modificado1['Medical Condition'] == valor]
  num_filas_a_eliminar2 = int(len(filas_objeto2) * porcentaje)
  filas_a_eliminar_aleatorias2 = filas_objeto2.sample(n=num_filas_a_eliminar2, random_state=1)
  indices_a_eliminar.extend(filas_a_eliminar2.index)

df_modificado2 = df_modificado1.drop(index=indices_a_eliminar)
'''
st.code(code, language='python')

# Paso 8: Redistribución de medicamentos
st.header('8. Redistribución de medicamentos')
st.write('Modificamos los medicamentos asignados a cada paciente en función de su edad y condición médica.')
code = '''
medications = {
    'Diabetes': {
        'meds': (['Metformin', 'Insulin', 'Glipizide'], [0.2, 0.7, 0.1]),
        'age_ranges': [(5, 40), (41, 60), (61, 100)]
    },
    'Arthritis': {
        'meds': (['Ibuprofen', 'Diclofenac', 'Methylprednisolone'], [0.5, 0.3, 0.2]),
        'age_ranges': [(31, 60), (5, 30), (61, 100)]
    }
}

for condition, info in medications.items():
    med_list, weights = info['meds']
    age_ranges = info['age_ranges']

    condition_df = new_df[new_df['Medical Condition'] == condition]

    for age_range in age_ranges:
        age_filtered_df = condition_df[(condition_df['Age'] >= age_range[0]) & (condition_df['Age'] <= age_range[1])]

        num_to_replace = int(len(age_filtered_df) * 1)

        if num_to_replace > 0:
            indices_to_replace = np.random.choice(age_filtered_df.index, num_to_replace, replace=False)
            new_medications = np.random.choice(med_list, num_to_replace, p=weights)
            new_df.loc[indices_to_replace, 'Medication'] = new_medications
'''
st.code(code, language='python')

# Finalización
st.success("¡Así se realizó la limpieza y ajuste de datos en nuestro dataset!")