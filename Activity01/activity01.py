import pandas as pd

data = pd.read_csv('AmesHousing.csv')

print(data.head())

print("\nInformation file:")
print(data.info())
