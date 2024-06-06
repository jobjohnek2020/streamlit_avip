import pandas as pd,numpy as np,seaborn as sns,matplotlib.pyplot as plt,streamlit as st
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

st.set_page_config(page_title='AIDS Virus Infection Prediction')
st.title('AIDS Virus Infection Prediction')

with st.expander('About'):
  st.markdown('What have I done in this app')
  st.info('Used dataset in kaggle https://www.kaggle.com/datasets/aadarshvelu/aids-virus-infection-prediction to predict AIDS infection')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Streamlit for user interface
  ''', language='markdown')



