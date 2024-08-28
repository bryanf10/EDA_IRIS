import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.datasets import load_iris

iris= load_iris()


##print (iris.keys())
##print (iris.target_names)

##print (iris.feature_names)
iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df["etiqueta"]=iris.target
##print(iris_df)


iris_df.describe()