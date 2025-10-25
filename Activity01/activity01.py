import pandas as pd

data = pd.read_csv('dataset_bolivia.csv')

print(data.head())

print("\nInformation file:")
print(data.info())
