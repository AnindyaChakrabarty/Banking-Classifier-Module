

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd

models = pickle.load(open('model.pkl','rb'))

def predictGUI(self,models,newDataList):
        import pandas as pd
        import numpy as np
        from statistics import mode
        newData_=pd.DataFrame(newDataList).T
        newData_.columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        processedDataHeaders=['CreditScore','Age','Tenure',	'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary','Geography_Germany','Geography_Missing','Geography_Spain','Gender_Male','Gender_Missing']
        newData_['CreditScore']=float(newData_['CreditScore'])
        newData_['Age']=float(newData_['Age'])
        newData_['Tenure']=int(newData_['Tenure'])
        newData_['Balance']=int(newData_['Balance'])
        newData_['NumOfProducts']=int(newData_['NumOfProducts'])
        newData_['HasCrCard']=int(newData_['HasCrCard'])
        newData_['IsActiveMember']=int(newData_['IsActiveMember'])
        newData_['EstimatedSalary']= float(newData_['EstimatedSalary'])
        
        newData_ = pd.get_dummies(newData_,drop_first=False)
        newData_= newData_.reindex(columns=processedDataHeaders,fill_value=0)
        finalPred=[]
        for key in models:
            print(key)
            finalPred.append(int(models[key].predict(newData_)))
            print(int(models[key].predict(newData_)))

        return mode(finalPred)

webapp = Flask(__name__)

@webapp.route('/')
def home():
    return render_template('index.html')


@webapp.route('/predict',methods=['POST'])
def predict():
    import pickle
    import pandas as pd
    import numpy as np
    from statistics import mode
    models = pickle.load(open('model.pkl','rb'))
    CreditScore=float(request.form.get('CreditScore'))
    Geography=request.form.get('Geography')
    Gender=request.form.get('Gender')
    Age=float(request.form.get('Age'))
    Tenure=int(request.form.get('Tenure'))
    Balance=int(request.form.get('Balance'))
    NumOfProducts=int(request.form.get('NumOfProducts'))
    HasCrCard=int(request.form.get('HasCrCard'))
    IsActiveMember=int(request.form.get('IsActiveMember'))
    EstimatedSalary=float(request.form.get('EstimatedSalary'))
    newDataList=[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]
    newData_=pd.DataFrame(newDataList).T
    newData_.columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    processedDataHeaders=['CreditScore','Age','Tenure',	'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary','Geography_Germany','Geography_Missing','Geography_Spain','Gender_Male','Gender_Missing']
    newData_ = pd.get_dummies(newData_,drop_first=False)
    newData_= newData_.reindex(columns=processedDataHeaders,fill_value=0)
    finalPred=[]
    for key in models:
        finalPred.append(key)
        finalPred.append(int(models[key].predict(newData_)))
        
    finalPred.append("Final Verdict :")
    finalPred.append(mode(finalPred)) 
    return render_template('index.html', prediction_text=finalPred)

if __name__ == "__main__":
    webapp.run(debug=True)

    

