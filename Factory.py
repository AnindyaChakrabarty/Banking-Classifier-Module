
from Classifier import Classifier, FeatureEngineering, ExploratoryDataAnalysis, Utility
from Input import Input ,MongoDB, Report

class App:

    def __init__(self,input):
        self.input_=input
    def fit(self):
        self.Classifier_=Classifier(self.input_)
        self.Classifier_.compareModel()
    def predict(self,newData):
        self.Classifier_.predict(newData)
    def saveResults(self):
        self.Classifier_.FE_.EDA_.logger_.debug("Saving Results in MongoDB")
        self.input_.writeMongoData(self.Classifier_.report_.report_,"ModelComparisonReport")
        self.input_.writeMongoData(self.Classifier_.predictionReport_.predictionReport_,"ModelPredictionReport")
        self.input_.writeMongoData(self.Classifier_.FE_.EDA_.missingList_,"MissingValueSummary")
        self.input_.writeMongoData(self.Classifier_.FE_.EDA_.cardinalityList_,"Cardinality")
        self.input_.writeMongoData(self.Classifier_.FE_.EDA_.catagoryList_,"FeatureTypes")
        self.Classifier_.FE_.EDA_.logger_.debug("***********************************************************")
        self.Classifier_.FE_.EDA_.logger_.debug("        Analysis Completed. thank you for your visit"       )
        self.Classifier_.FE_.EDA_.logger_.debug("***********************************************************")
        

