import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Carga el archivo CSV usando Pandas
df = pd.read_csv('df0.csv')

# Título de la app
st.title("Análisis de datos hospitalarios")

# Estilo para subtítulos
st.markdown("""

""", unsafe_allow_html=True)

# Sección 1: Distribución de condiciones médicas
st.header("Distribución de condiciones médicas")
conteos = df['Medical Condition'].value_counts()
porcentajes1 = (conteos / len(df)) * 100
df_porcentaje = porcentajes1.reset_index()
df_porcentaje.columns = ['Condition', 'Percentage']
total_filas = len(df)

grafico = alt.Chart(df_porcentaje).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field='Percentage', type='quantitative', title='Percentage'),
    color=alt.Color(field='Condition', type='nominal', title='Condition'),
    tooltip=[alt.Tooltip(field='Condition', type='nominal', title='Condition'),
             alt.Tooltip(field='Percentage', type='quantitative', title='Percentage')]
).properties(
    title='Distribución de condiciones médicas',
    width=400,
    height=400
)

df_texto_total = pd.DataFrame({
    'x': [0],
    'y': [0],
    'texto': [f'Número total de pacientes: {total_filas}']
})

texto_total = alt.Chart(df_texto_total).mark_text(
    align='center',
    baseline='middle',
    fontSize=18,
    dy=180,  # Ajuste para acercar el texto al gráfico
    color='white'  # Cambiar a color blanco
).encode(
    x=alt.X('x:O', axis=None),
    y=alt.Y('y:O', axis=None),
    text='texto:N'
).properties(
    width=400,
    height=400
)

grafico_final = grafico + texto_total
st.altair_chart(grafico_final, use_container_width=True)

# Sección 2: Medicamentos Más Utilizados por Condición
st.markdown("Medicamentos más utilizados por condición", unsafe_allow_html=True)

def graficar_medicamentos(condition):
    df_condition = df[df['Medical Condition'] == condition]
    conteos_medicamentos = df_condition['Medication'].value_counts()
    total_condition = len(df_condition)
    top_3_medicamentos = conteos_medicamentos.head(3).reset_index()
    top_3_medicamentos.columns = ['Medication', 'Count']

    grafico_barras = alt.Chart(top_3_medicamentos).mark_bar().encode(
        x=alt.X('Medication:O', title='Medication'),
        y=alt.Y('Count:Q', title='Número de usos', axis=alt.Axis(format='d')),
        color='Medication:N',
        tooltip=[alt.Tooltip(field='Medication', type='nominal', title='Medication'),
                 alt.Tooltip(field='Count', type='quantitative', title='Número de usos')]
    ).properties(
        title=f'Tres medicamentos más usados para "{condition}"',
        width=400,
        height=300
    )

    df_texto_total = pd.DataFrame({
        'x': [0],
        'y': [0],
        'texto': [f'Pacientes con "{condition}": {total_condition}']
    })

    texto_total = alt.Chart(df_texto_total).mark_text(
        align='center',
        baseline='bottom',
        fontSize=18,
        dy=160,  # Ajuste para acercar el texto al gráfico
        color='white'  # Cambiar a color blanco
    ).encode(
        x=alt.X('x:O', axis=None),
        y=alt.Y('y:O', axis=None),
        text='texto:N'
    ).properties(
        width=400,
        height=300
    )

    grafico_final = grafico_barras + texto_total
    return grafico_final

conditions = ["Asthma", "Diabetes", "Arthritis"]
for condition in conditions:
    st.subheader(condition)
    st.altair_chart(graficar_medicamentos(condition), use_container_width=True)

# Sección 3: Gasto Promedio por Estancia Hospitalaria
st.markdown("Gasto promedio por estancia hospitalaria", unsafe_allow_html=True)

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Hospitalization Days'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

for condition in conditions:
    df_condition = df[df['Medical Condition'] == condition]
    average_billing_amount = int(df_condition['Billing Amount'].mean())
    average_hospitalization_days = df_condition['Hospitalization Days'].mean()
    st.write(f'**Días promedio de hospitalización para pacientes con {condition}:** {average_hospitalization_days:.2f} días', unsafe_allow_html=True)
    st.write(f'**Costo promedio de hospitalización para {condition}:** {average_billing_amount}', unsafe_allow_html=True)

# Sección 4: Gráfico de Aseguradoras
st.markdown("Distribución de aseguradoras", unsafe_allow_html=True)

conteo_aseguradoras = df['Insurance Provider'].value_counts()
porcentaje_aseguradoras = (conteo_aseguradoras / total_filas) * 100
resultado = pd.DataFrame({
    'Insurance Provider': porcentaje_aseguradoras.index,
    'Percentage': porcentaje_aseguradoras.values
}).sort_values(by='Percentage', ascending=False)

st.write(resultado)

aseguradora = 'UnitedHealthcare'
df_aseguradora = df[df['Insurance Provider'] == aseguradora]
conteo_condiciones = df_aseguradora['Medical Condition'].value_counts().reset_index()
conteo_condiciones.columns = ['Medical Condition', 'Count']
total_pacientes_aseguradora = len(df_aseguradora)
conteo_condiciones['Percentage'] = (conteo_condiciones['Count'] / total_pacientes_aseguradora) * 100

chart = alt.Chart(conteo_condiciones).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field='Percentage', type='quantitative'),
    color=alt.Color(field='Medical Condition', type='nominal'),
    tooltip=['Medical Condition', 'Count', 'Percentage:Q']
).properties(
    title=f'Distribución de condiciones médicas cubiertas por {aseguradora}'
)

text = alt.Chart(pd.DataFrame({
    'text': [f'Total de pacientes asegurados: {total_pacientes_aseguradora}']
})).mark_text(
    align='center',
    baseline='middle',
    fontSize=18,
    dy=160,
    color='white'  # Cambiar a color blanco
).encode(
    text='text:N'
)

final_chart = alt.layer(chart, text).properties(
    height=350
)
st.altair_chart(final_chart, use_container_width=True)

# Sección 5: Mediana de Edad
st.markdown("Mediana de edad por condición médica", unsafe_allow_html=True)

mediana_age = df.groupby('Medical Condition')['Age'].median().reset_index()
mediana_age.columns = ['Medical Condition', 'Median Age']

chart = alt.Chart(mediana_age).mark_bar().encode(
    x=alt.X('Medical Condition', sort='-y'),
    y=alt.Y('Median Age:Q', title='Edad Mediana'),
    color='Medical Condition:N'
).properties(
    title='Edad Mediana por Condición Médica',
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)