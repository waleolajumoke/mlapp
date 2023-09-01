import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import matplotlib
matplotlib.use('Agg')

from PIL import Image
image = Image.open('logo.png')
st.image(image,use_column_width=True)
st.title("Heart Disease Prediction")
# st.text_input("Enter your age")
st.write("Let's explore different classifiers and dataset")

dataset_name = st.sidebar.selectbox("Select dataset", ("Breast Cancer","Iris","Wine"))
classifier_name = st.sidebar.selectbox("Select Classifier",("SVM","KNN"))

def get_dataset(name):
    data=None
    if name =="Iris":
        data=datasets.load_iris()
    elif name =="Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target

    return x,y 

x,y=get_dataset(dataset_name)
st.dataframe(x)
st.write(x.shape)
