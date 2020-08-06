
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 


from Classifier import Classifier, FeatureEngineering, ExploratoryDataAnalysis, Utility
from Input import Input ,MongoDB, Report



class App:

    def __init__(self,input):
        self.input_=input
        
               

    def runClassifier(self):
        self.Classifier=Classifier(self.input_)
        self.Classifier.compareModel()
        self.input_.writeMongoData(app.Classifier.report_.report_,"ModelComparisonReport")




if __name__=='__main__':
    
    inp=Input("BankChurnData","RawData","Exited")
    app=App(inp)                          
    app.runClassifier()









   





