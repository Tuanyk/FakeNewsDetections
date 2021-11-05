import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('news.csv')
df.loc[df.label == 'FAKE', 'label'] = 0
df.loc[df.label == 'REAL', 'label'] = 1

