import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score



st.set_page_config(page_title='Breast Cancer Detection', page_icon='breastcancer_pic.jpg', layout='centered', initial_sidebar_state='auto')
    
st.title('Breast Cancer Detection')
st.image('breastcancer_pic.jpg',width=150)
st.sidebar.title('Breast Cancer Detection')
st.markdown('Cancer is Malignant or Benign? ')
navigation=st.sidebar.radio('VIEW', ('Data Analysis','Prediction'))


data=pd.read_csv("breast cancer data 1.csv")
data=data.drop(columns=["id"])
data=data.drop(columns=["Unnamed: 32"])
df=pd.read_csv("cleaned_breastcancer_data.csv")
df=df.drop(columns=["Unnamed: 0"])

@st.cache(persist=True)
def split(df):
    y = df['diagnosis']
    
    prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean','radius_worst','perimeter_worst',	'area_worst',	'concave points_worst', 'concave points_mean']
    train, test = train_test_split(df, test_size = 0.3,random_state=3)
    x_train = train[prediction_feature]
    y_train = train.diagnosis

    x_test = test[prediction_feature]
    y_test = test.diagnosis
    
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RS)
    return x_train, x_test, y_train, y_test








if navigation =="Data Analysis":


     if st.sidebar.checkbox("Show Raw Data", False):
            st.subheader('Breast Cancer Dataset')
            #st.write(data.head(100))
            st.dataframe(data)


     if st.sidebar.checkbox("Show Cleaned Data",False):
         st.subheader("Cleaned Breast Cancer Dataset")
         #st.write(df.head(100))
         st.dataframe(df)

     plots=st.sidebar.multiselect("Plots",('Scatter Matrix', 'Number of Malignant and Benign','Heatmap','Mean radius vs Mean area','Worst Concave Points  vs Worst Area'))


     if st.sidebar.button("Plot", key='plotss'):
            with st.spinner('Wait for it...'):
                time.sleep(5)
     
    
     if 'Number of Malignant and Benign' in plots:
                st.subheader("Malignant and Benign Count")
                fig,ax = plt.subplots()
                ma = len(df[df['diagnosis']==1])
                be = len(df[df['diagnosis']==0])
                count=[ma,be]
                bars = plt.bar(np.arange(2), count, color=['yellowgreen','salmon'])
                for bar in bars:
                    height = bar.get_height()
                    plt.gca().text(bar.get_x() + bar.get_width()/2, height*.80, '{0:.{1}f}'.format(height, 1), ha='center', color='black', fontsize=8)
                plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                plt.xticks(ticks=[0,1])
                ax.set_ylabel('Count')
                ax.set_xlabel('Target')
                ##remove dashes from frame
                ax.xaxis.set_tick_params(length=0)
                ax.yaxis.set_tick_params(length=0)
                st.pyplot(fig)
     if 'Scatter Matrix' in plots:
                st.subheader("Scatter Matrix")
                fig = px.scatter_matrix(df,dimensions=["radius_mean" ,'perimeter_mean','area_mean','concave points_mean',"radius_worst"],color="diagnosis",width = 800,height = 700)
                st.write(fig)
            
     if 'Heatmap' in plots:
                st.subheader("Heatmap")
                fig=plt.figure(figsize = (30,20))
                hmap=sns.heatmap(df.drop(columns=['diagnosis']).corr(), annot = True,cmap= 'Blues',annot_kws={"size": 30})
                hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize = 17)
                hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize = 17)
                st.pyplot(fig)
     if 'Mean radius vs Mean area' in plots:
                st.subheader('Cancer Radius and Area')
                fig = plt.figure()
                sns.scatterplot(x=df['radius_mean'],y = df['area_mean'],hue = df['diagnosis'],palette=['#000099','#ffff00'])
                st.pyplot(fig)
     if 'Worst Concave Points  vs Worst Area' in plots:
                st.subheader('Cancer Worst Concave Points  and Worst Area')
                fig = plt.figure()
                sns.scatterplot(x=df['concave points_worst'],y = df['area_worst'],hue = df['diagnosis'],palette=['#000099','#ffff00'])
                st.pyplot(fig)




###---------------classification----------------------
if navigation == 'Prediction':

            x_train, x_test, y_train, y_test = split(df)
            if st.sidebar.checkbox("Show X_train/Y_train", False):
                          st.subheader('X_train')
                          st.dataframe(x_train)
                          st.subheader('Y_train')
                          st.dataframe(y_train)

            st.sidebar.subheader("Classifier: Logistic Regression")
            st.sidebar.subheader("Model Hyperparameters")
    
            c = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
            

           
            prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean','radius_worst','perimeter_worst',	'area_worst',	'concave points_worst', 'concave points_mean']
                        
    
            def user_inputs_ui(df):
                user_val = {}
                X = df.drop(["diagnosis"], axis=1)
                for col in X.columns:
                      name=col
                      col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
                      user_val[name] = round((col),4)
                return user_val
            user_val=user_inputs_ui(df)

            def user_predict():
                 global U_pred
                 X= df.drop(["diagnosis"], axis=1)
                 model = LogisticRegression(C=c, penalty='l2', max_iter=max_iter)
                 model.fit(x_train, y_train)
                 accuracy = model.score(x_test, y_test)
                 y_pred = model.predict(x_test)
                 
                 U_pred = model.predict([[user_val[col] for col in X.columns]])
                 st.subheader("Your Status: ")
                 if U_pred == 0:
                        st.write(U_pred[0], " - You are not at high risk :)")
                 else:
                        st.write(U_pred[0], " - You are at high risk :(")
                 
                 class_names = ['malignant', 'benign']
                 st.write("Accuracy: ", accuracy.round(2))
                 st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                 st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                

                       
            user_predict()  #Predict the status of user.


                           
                                      
            
    




































                







                
                




















    
