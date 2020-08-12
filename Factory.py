
from Classifier import Classifier, FeatureEngineering, ExploratoryDataAnalysis, Utility
from Input import Input ,MongoDB, Report

class App:

    def __init__(self,input):
        self.input_=input
        
      
    def fit(self):
        self.Classifier_=Classifier(self.input_)
        self.Classifier_.compareModel()
        self.input_.writeMongoData(self.Classifier_.report_.report_,"ModelComparisonReport")
    def predict(self,newData):
        self.Classifier_.predict(newData)
        self.input_.writeMongoData(self.Classifier_.predictionReport_.predictionReport_,"ModelPredictionReport")

