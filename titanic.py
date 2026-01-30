import pandas as pd

df = pd.read_csv('train.csv')

print("--- 5 Filas ---")
print(df.head(5))

print("\n--- Información General ---")
print(df.info())

print("\n--- Resumen Estadístico ---")
print(df.describe())