import pandas as pd
import numpy as np

df = pd.DataFrame(index=None)

df['FC'] = None
df['fALFF'] = None
df['Reho'] = None
df['Status'] = None

df['FC'] = [0.1, 0.2, 0.3, 0.4, 0.5]
df['fALFF'] = [0.6, 0.7, 0.8, 0.9, 1.0]
df['Reho'] = [1.1, 1.2, 1.3, 1.4, 1.5]
df['Status'] = ['Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy']

print(df.head())