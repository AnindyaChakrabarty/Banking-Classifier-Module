
from Classifier import Classifier, FeatureEngineering, ExploratoryDataAnalysis, Utility
from Input import Input ,MongoDB, Report

class App:

    def __init__(self,input):
        self.input_=input
      
    def runClassifier(self):
        self.Classifier_=Classifier(self.input_)
        self.Classifier_.compareModel()
        self.input_.writeMongoData(self.Classifier_.report_.report_,"ModelComparisonReport")