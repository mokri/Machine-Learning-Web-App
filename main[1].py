import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


@st.cache(persist=True)
def load_data():
    df = pd.read_csv('diabete.csv')

    #labeling = LabelEncoder()
    #for col in df.columns:
    #    df[col] = labeling.fit_transform(df[col])
    return df


@st.cache(persist=True)
def split_data(df):
    y = df.Outcome
    x = df.drop(columns=['Outcome'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test


def plot_matrix(matrix_list, model):
    if 'Confusion Matrix' in matrix_list:
        st.subheader('Confusion Matrix')
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in matrix_list:
        st.subheader('ROC Curve')
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision Recall Curve' in matrix_list:
        st.subheader('Precision Recall Curve')
        plot_precision_recall_curve(model, x_train, y_train)

        st.pyplot()


def svm():
    st.sidebar.subheader('Model Hyperparameters')
    c = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, key='C')
    kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key='K')
    gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key='G')
    mertics = st.sidebar.multiselect('What metrics to plot?',
                                     ('Confusion Matrix', 'ROC Curve', 'Precision Recall Curve'))

    if st.sidebar.button('Classify', key='classify'):
        st.subheader('Suppor Vector Machine (SVM) Results:')
        model = SVC(C=c, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write('Accuracy', accuracy.round(2))
        st.write('Precision', precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write('Recall', recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_matrix(mertics, model)


def logistic_regression():
    st.sidebar.subheader('Model Hyperparameters')
    c = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, key='C_LR')
    max_iter = st.sidebar.slider('Maximum number of iterations', 100, 1000, key='max_iter')
    mertics = st.sidebar.multiselect('What metrics to plot?',
                                     ('Confusion Matrix', 'ROC Curve', 'Precision Recall Curve'))

    if st.sidebar.button('Classify', key='classify'):
        st.subheader('Logistic Regression Results:')
        model = LogisticRegression(C=c, max_iter=max_iter, )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write('Accuracy', accuracy.round(2))
        st.write('Precision', precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write('Recall', recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_matrix(mertics, model)


def random_forest():
    st.sidebar.subheader('Model Hyperparameters')
    n_estimators = st.sidebar.number_input('The number of trees in the forest', 100, 5000, step=10, key='n_estimators')
    max_depth = st.sidebar.number_input('The maximum depth of the tree', 1, 20, step=1, key='max_depth')
    bootstrap = st.sidebar.radio('Bootstrap samples when building trees', ('True', 'False'), key='bootstrap')
    mertics = st.sidebar.multiselect('What metrics to plot?',
                                     ('Confusion Matrix', 'ROC Curve', 'Precision Recall Curve'))

    if st.sidebar.button('Classify', key='classify'):
        st.subheader('Random Tree Results:')
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write('Accuracy', accuracy.round(2))
        st.write('Precision', precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write('Recall', recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_matrix(mertics, model)
        

if __name__ == '__main__':

    st.title('Binary Classification Web App')
    st.sidebar.title('Options')
    st.markdown('is the patient diabetic or not ?')

    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ['not diabetic', 'diabetic']
    st.sidebar.subheader('Choose Classifier')

    classifier = st.sidebar.selectbox('Classifier',
                                      ('Support Vector Machine(SVM)', 'Logistic Regression', 'Random Forest'))

    if classifier == 'Support Vector Machine(SVM)':
        svm()
    elif classifier == 'Logistic Regression':
        logistic_regression()
    elif classifier == 'Random Forest':
        random_forest()
    if st.sidebar.checkbox('Show raw data', False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
