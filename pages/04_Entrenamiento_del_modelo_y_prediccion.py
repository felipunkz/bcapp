import streamlit as st 
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

# Definir los modelos para evitar reentrenarlos cada vez 
stacking_clf = None 
 
if 'clean_df' in st.session_state: 
    # df = pd.read_csv(uploaded_file) 
 
    # # Mostrar primeras filas del DataFrame 
    # st.write("Vista previa de los datos:") 
    # st.write(df.head(3)) 
 
    # # Selecciona solo las columnas que deseas conservar 
    # df = df[['Age', 'Medication', 'Medical Condition']] 
 
    # # Obtener los 5 medicamentos más comunes 
    # top_medications = df['Medication'].value_counts().index[:10] 
 
    # # Filtrar el DataFrame para incluir solo esos medicamentos 
    # df_reduced = df[df['Medication'].isin(top_medications)] 
 
    # # Preparar los datos 
    # X = df_reduced[['Age', 'Medical Condition']] 
    # y = df_reduced['Medication'] 

    df = st.session_state['clean_df']
    X = st.session_state['reduced_attr']
    y = st.session_state['reduced_tgt']

    label_encoder = LabelEncoder() 
    y_encoded = label_encoder.fit_transform(y) 
    y_encoded_cat = to_categorical(y_encoded) 

    X_numeric = X[['Age']].values 
    X_categorical = X[['Medical Condition']].values 

    scaler = StandardScaler() 
    X_numeric_scaled = scaler.fit_transform(X_numeric) 

    encoder = OneHotEncoder() 
    X_categorical_encoded = encoder.fit_transform(X_categorical).toarray() 

    X_prepared = np.concatenate([X_numeric_scaled, X_categorical_encoded], axis=1) 

    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_encoded, test_size=0.4, random_state=42) 

    # Modelo de red neuronal 
    def create_nn_model(input_dim): 
        model = Sequential() 
        model.add(Input(shape=(input_dim,)))  # Se reemplaza input_dim con Input 
        model.add(Dense(128, activation='relu')) 
        model.add(Dropout(0.3)) 
        model.add(Dense(64, activation='relu')) 
        model.add(Dense(len(label_encoder.classes_), activation='softmax')) 
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model 

    class KerasModelWrapper(BaseEstimator, ClassifierMixin): 
        def __init__(self, epochs=10, batch_size=32): 
            self.epochs = epochs 
            self.batch_size = batch_size 
            self.model = None 
            self.classes_ = None 

        def fit(self, X, y): 
            self.classes_ = np.unique(y) 
            self.model = create_nn_model(X.shape[1]) 
            try: 
                self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0) 
            except Exception as e: 
                st.error(f"Error durante el entrenamiento: {e}") 
            return self 

        def predict(self, X): 
            try: 
                predictions = self.model.predict(X) 
                return np.argmax(predictions, axis=1) 
            except Exception as e: 
                st.error(f"Error en la predicción: {e}") 
                return np.array([]) 

        def predict_proba(self, X): 
            return self.model.predict(X) 

    # Solo entrenar una vez 
    if stacking_clf is None: 
        rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_split=2) 
        nn_model = KerasModelWrapper(epochs=10, batch_size=32) 

        estimators = [ 
            ('rf', rf), 
            ('nn', nn_model) 
        ] 

        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

        stacking_clf.fit(X_train, y_train) 

        stacking_accuracy = stacking_clf.score(X_test, y_test) 
        st.write("Precisión del modelo de Stacking en el conjunto de prueba:", stacking_accuracy) 

    # Sección interactiva para predicción 
    st.subheader("Predice el medicamento") 
    with st.form("my_form"):
        st.write("Predicción de medicamento")
        age_input = st.number_input("Edad del paciente", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), step=1) 
        condition_input = st.selectbox("Condición Médica", df['Medical Condition'].unique()) 
        # st.form_submit_button(form_button_label, disabled=form_button_disabled, type='primary', on_click=predicts, args=[age_input,condition_input])
        st.form_submit_button("Enviar datos",  type='primary')
        # This is outside the form
        st.write(age_input)
        st.write(condition_input)
    # Inputs del usuario 
    # age_input = st.number_input("Edad del paciente", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), step=1) 
    # condition_input = st.selectbox("Condición Médica", df['Medical Condition'].unique()) 

    # if st.button("Predecir Medicamento"): 
    #     # Preparar la entrada del usuario 
        try: 
            age_scaled = scaler.transform([[age_input]]) 
            condition_encoded = encoder.transform([[condition_input]]).toarray() 
            user_input = np.concatenate([age_scaled, condition_encoded], axis=1) 

            # Predicción 
            pred = stacking_clf.predict(user_input) 
            predicted_medication = label_encoder.inverse_transform(pred) 

            # Mostrar el resultado 
            st.write(f"El medicamento recomendado es: {predicted_medication[0]}") 
        except Exception as e: 
            st.error(f"Error durante la predicción: {e}")


