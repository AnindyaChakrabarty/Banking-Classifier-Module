
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 


from Classifier import Classifier, FeatureEngineering, ExploratoryDataAnalysis, Utility
from Input import Input ,MongoDB, Report
from Factory import App
import pickle




inp=Input("BankChurnData","RawData","Exited")
app=App(inp)                          
app.fit()
newDataList=[619, 'France', 'Female', 42, 2, 0, 1, 1, 1, 101348.88]
app.predictGUI(newDataList)
app.predict("newData")
app.saveResults()









   





