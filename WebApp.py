


from flask import Flask, request, jsonify, render_template


webapp = Flask(__name__)


@webapp.route('/')
@webapp.route('/home')
def home():
    return render_template('/home.html')


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
    table.append(f'{finalPred[0]} : {finalPred[1]}')
    table.append(f'{finalPred[2]} :{finalPred[3]}')
    table.append(f' {finalPred[4]} :{finalPred[5]}')
    table.append(f'{finalPred[6]}:{finalPred[7]}')
    table.append(f'{finalPred[8]} :{finalPred[9]}')
    table.append(f'{finalPred[10]} :{finalPred[11]}')
    table.append(f'{finalPred[12]} :{finalPred[13]}')
    table.append(f'{finalPred[14]} : {finalPred[15]}')
    table.append(f'{finalPred[16]} : {finalPred[17]}')
      
    
    return render_template('home.html', prediction_text=table)



if __name__ == "__main__":
    webapp.run(debug=True)

    

