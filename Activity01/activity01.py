import pandas as pd

data = pd.read_csv('cybersecurity_attacks.csv')

print(data.head())

print("\nInformation file:")
print(data.info())
