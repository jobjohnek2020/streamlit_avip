import pandas as pd,numpy as np,seaborn as sns,matplotlib.pyplot as plt,streamlit as st
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='AIDS Virus Infection Prediction')
st.title('AIDS Virus Infection Prediction')

with st.expander('About'):
  st.info('Used dataset in kaggle to predict AIDS infection')
  st.link_button("AIDS Infection Prediction Kaggle Link", 'https://www.kaggle.com/datasets/aadarshvelu/aids-virus-infection-prediction')
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Streamlit for user interface
  ''', language='markdown')

# reading dataset
df = pd.read_csv('dataset/AIDS_Classification_50000.csv')

# Number of input features
x_variables = len(df.columns) - 1

# removing outliers
Q1 = df.quantile(q=0.25)
Q3 = df.quantile(q=0.75)
IQR = df.apply(stats.iqr)
data_clean = df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]

# removing skewness
x = data_clean.iloc[:,:-1]
y = data_clean.iloc[:,-1].values
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=1)
x_t = quantile_transformer.fit_transform(x.values)

# splitting data as training and testing data
x_train,x_test,y_train,y_test = train_test_split(x_t,y,test_size=0.30,random_state=1)

# setting input data in UI
st.header('Input data', divider='rainbow')
col = st.columns(4)
col[0].metric(label='No of samples',value=len(data_clean),delta='')
col[1].metric(label='No of X variables',value=x_variables,delta='')
col[2].metric(label='No of training samples',value=x_train.shape[0],delta='')
col[3].metric(label='No of testing samples',value=x_test.shape[0],delta='')

# setting initial dataset in UI
with st.expander('Initial dataset',expanded=True):
  st.dataframe(df.head(10),height=210,use_container_width=True)

# setting training dataset in UI
with st.expander('Training split',expanded=True):
  train_col = st.columns((3,1))
  with train_col[0]:
    st.markdown('X')
    st.dataframe(x_train,height=210,use_container_width=True)
  with train_col[1]:
    st.markdown('Y')
    st.dataframe(y_train,height=210,use_container_width=True)

# setting testing dataset in UI
with st.expander('Testing split',expanded=True):
  test_col = st.columns((3,1))
  with test_col[0]:
    st.markdown('X')
    st.dataframe(x_test,height=210,use_container_width=True)
  with test_col[1]:
    st.markdown('Y')
    st.dataframe(y_test,height=210,use_container_width=True)

# Models and prediction
models = [KNeighborsClassifier(n_neighbors=13),BernoulliNB(),SVC(),DecisionTreeClassifier(criterion='entropy')]
scores = []
for model in models:
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  score = accuracy_score(y_test,y_pred)
  scores.append(score)
model_df = pd.DataFrame(
  data = {
    "Name" : ['KNeighborsClassifier','BernoulliNB','SVC','DecisionTreeClassifier'],
    "Accuracy Score" : [scores[0],scores[1],scores[2],scores[3]]
  },
  index = [1,2,3,4]
)
st.header('Models and results',divider='rainbow')
st.dataframe(model_df,height=210,use_container_width=True)
