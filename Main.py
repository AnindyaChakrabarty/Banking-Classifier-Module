
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 


from Classifier import Classifier, FeatureEngineering, ExploratoryDataAnalysis, Utility
from Input import Input ,MongoDB, Report
from Factory import App



inp=Input("BankChurnData","RawData","Exited")
app=App(inp)                          
app.runClassifier()




    
    









   





