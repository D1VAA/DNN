import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

data = pd.read_excel('./PGTO-MOT.xlsx')
encoder = OneHotEncoder()

colunas_one_hot = ['Origem', 'UF', 'Destino', 'UF2', 'Propriedade']
X_categorico = data[colunas_one_hot] 
X_numerico = data[['Percurso', 'Peso (kg)']]
y = data['Valor Pago']

transformador = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), colunas_one_hot),
        ('scaler', StandardScaler(), X_numerico.columns)
    ],
    remainder='passthrough'
)
X_transformado = transformador.fit_transform(data.drop('Valor Pago', axis=1))

X_train, X_test, y_train, y_test = train_test_split(X_transformado, y, test_size=0.2, random_state=42)
X_train = X_train.toarray()
X_test = X_test.toarray()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='sigmoid', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(units=32, activation='sigmoid'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=4000, batch_size=32, validation_data=(X_test, y_test))
model.save('modelo.h5')

loss = model.evaluate(X_test, y_test)
print('\nLoss no conjunto de testes: ', loss)

previsao = model.predict(X_test)