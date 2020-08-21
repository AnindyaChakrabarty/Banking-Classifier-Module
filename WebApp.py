


from flask import Flask, request, jsonify, render_template


webapp = Flask(__name__)


@webapp.route('/')
@webapp.route('/home')
def home():
    return render_template('/ClassifyForm.html')


@webapp.route('/predict',methods=['POST'])
def predict():
    import pickle
    import pandas as pd
    import numpy as np
    from statistics import mode
    from astropy.table import QTable, Table, Column
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
        if int(models[key].predict(newData_))==1:
            verdict="Customer will Exit"
        else:
            verdict ="Customer will not Exit"
        finalPred.append(verdict)
        
    finalPred.append("Final Verdict :")
    finalPred.append(mode(finalPred))
    table=[]
    table.append(f'Prediction using {finalPred[0]} is that {finalPred[1]}')
    table.append(f'Prediction using {finalPred[2]} is that {finalPred[3]}')
    table.append(f'Prediction using {finalPred[4]} is that {finalPred[5]}')
    table.append(f'Prediction using {finalPred[6]} is that {finalPred[7]}')
    table.append(f'Prediction using {finalPred[8]} is that {finalPred[9]}')
    table.append(f'Prediction using {finalPred[10]} is that {finalPred[11]}')
    table.append(f'Prediction using {finalPred[12]} is that {finalPred[13]}')
    table.append(f'Prediction using {finalPred[14]} is that {finalPred[15]}')
    table.append(f'Final Prediction from this Model is that {finalPred[17]}')
        
    return render_template('Output.html', prediction_text=table)



if __name__ == "__main__":
    webapp.run(debug=True)

    

