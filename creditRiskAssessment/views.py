from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')

def result(request):
    person_age = int(request.GET.get('n1'))
    person_income = int(request.GET.get('n2'))
    person_home_ownership = str(request.GET.get('n3'))
    person_emp_length = int(request.GET.get('n4'))
    loan_intent = str(request.GET.get('n5'))
    loan_grade = str(request.GET.get('n6'))
    loan_amnt = int(request.GET.get('n7'))
    loan_int_rate = int(float(request.GET.get('n8')) * 100)  # Convert float to integer (percentage)
    loan_percent_income = int(float(request.GET.get('n9')) * 100)  # Convert float to integer (percentage)
    cb_person_default_on_file = str(request.GET.get('n10'))
    cb_person_cred_hist_length = int(request.GET.get('n11'))

    data = pd.read_csv('creditRiskAssessment/credit_risk_dataset.csv')
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    lone_dict = {}

    for i, j in enumerate(data['loan_intent'].unique()):
        lone_dict[j] = i
    data['loan_intent'] = data['loan_intent'].map(lone_dict)

    person_home_owner_dict = {}

    for i, j in enumerate(data['person_home_ownership'].unique()):
        person_home_owner_dict[j] = i
    data['person_home_ownership'] = data['person_home_ownership'].map(person_home_owner_dict)

    le=LabelEncoder()
    le_loan_grade = le.fit(data['loan_grade'])
    data['loan_grade'] = le_loan_grade.transform(data['loan_grade'])
    loan_grade_dict = dict()
    for i in le_loan_grade.classes_:
        loan_grade_dict[i] = le_loan_grade.transform([i])[0]

    le_cb_person_default_on_file = le.fit(data['cb_person_default_on_file'])
    data['cb_person_default_on_file'] = le_cb_person_default_on_file.transform(data['cb_person_default_on_file'])
    cb_person_default_on_file_dict = dict()
    for i in le_cb_person_default_on_file.classes_:
        cb_person_default_on_file_dict[i] = le_cb_person_default_on_file.transform([i])[0]

    x = data.drop(['loan_status'], axis=1)
    y = data['loan_status']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    def predict_the(lone_dict, loan_grade_dict, person_home_owner_dict, cb_person_default_on_file_dict, person_age,
                    person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt,
                    loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length):
        loan_intent = lone_dict[loan_intent]
        loan_grade = loan_grade_dict[loan_grade]
        person_home_ownership = person_home_owner_dict[person_home_ownership]
        cb_person_default_on_file = cb_person_default_on_file_dict[cb_person_default_on_file]

        input_data = pd.DataFrame({
            'person_age': [person_age],
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership],
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_default_on_file': [cb_person_default_on_file],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length]
        })

        prediction = rfc.predict(input_data)
        return prediction

    r = predict_the(lone_dict, loan_grade_dict, person_home_owner_dict, cb_person_default_on_file_dict, person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade, loan_amnt,loan_int_rate,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length)

    if r == 0:
        p="Loan may not be sanctioned"
    else:
        p="Loan might be sanctioned"

    return render(request, 'predict.html', {"result": p})