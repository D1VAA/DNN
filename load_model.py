import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

model = tf.keras.models.load_model('modelo.h5')
data = pd.read_excel('./conjunto_teste.xlsx')
colunas_one_hot = ['Origem', 'UF', 'Destino', 'UF2', 'Propriedade']
X_categorico = data[colunas_one_hot] 
X_numerico = data[['Percurso', 'Peso (kg)']]

transformador = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), colunas_one_hot),
        ('scaler', StandardScaler(), X_numerico.columns)
    ],
    remainder='passthrough'
)

data_transformado = transformador.fit_transform(data)

previsoes = model.predict(data_transformado)

data['Valor'] = previsoes

data.to_excel('Resultado.xlsx', index=False)
