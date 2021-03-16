import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('my_tipdata.csv')
pd.DataFrame(df,columns=['total_bill','sex','smoker','day','time','size','tip'])
df.sex = df.sex.astype('category')

X = df[['total_bill','sex','size']] 
Y = df['tip']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))



