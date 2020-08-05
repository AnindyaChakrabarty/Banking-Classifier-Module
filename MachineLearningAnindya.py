
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 
# The different classes developed follows the lyfecycle of a datascience projects 
# Classes used : 
# Exploratory Data Analysis (EDA) 
# Feature Engineering 
# Feature Selection 
# Model 
# Deployment 



class Utility:
    
    
    def makeDir(self,parentDirectory, dirName):
        import os
        new_dir = os.path.join(parentDirectory, dirName+'\\')
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        return new_dir
        


class ExploratoryDataAnalysis:
    
    def __init__(self,dataFileName,dependentVariableName):
        import pandas as pd
        import os
        import numpy as np
        logger.debug("Starting Exploratory Data Analysis")
        self.util_=Utility()
        self.dataFileName_=dataFileName
        self.dependentVariableName_=dependentVariableName
        self.dataset_= pd.read_csv(dataFileName)
        self.parentDirectory_ = os.path.dirname(os.getcwd())
        self.exploratoryDataAnalysisDir_= self.util_.makeDir(self.parentDirectory_,"Exploratory Data Analysis")
        self.writer_ = pd.ExcelWriter(self.exploratoryDataAnalysisDir_+'Exploratory Data Analysis.xlsx', engine='xlsxwriter')
    
                
    def getY(self):
        self.Y_=self.dataset_[self.dependentVariableName_]
    def getX(self):
        Header=list(self.dataset_.columns)
        Header.remove(self.dependentVariableName_)
        self.X_=self.dataset_[Header]
    def getData(self):
        Header=list(self.dataset_.columns)
        Header.remove('RowNumber')
        Header.remove('CustomerId') 
        self.data_=self.dataset_[Header]
    def getMissingValues(self):
        import pandas as pd
        import os
        import numpy as np 
        import matplotlib.pyplot as plt
        import seaborn as sns
       
        Missing_Features_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Missing Features")
        Missing_Features_Plot_Dir= self.util_.makeDir(Missing_Features_Dir,"Missing Features Plot")
        data=self.data_.copy()
        self.features_With_NA_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1]
        self.missing_Continuous_Numerical_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1 and self.data_[feature].dtype!="O" and feature !="Id" and len(self.data_[feature].unique())>25]
        self.missing_Discrete_Numerical_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1 and self.data_[feature].dtype!="O" and feature !="Id" and len(self.data_[feature].unique())<=25]
        self.missing_Catagorical_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1 and self.data_[feature].dtype=="O"]

        self.missingList_= pd.DataFrame(columns=['Features','Total Missing Values','% Missing Values'])
        for feature in self.features_With_NA_:
            self.missingList_ = self.missingList_.append({'Features': feature,'Total Missing Values': np.round(self.data_[feature].isnull().sum(),0), '% Missing Values': np.round(self.data_[feature].isnull().mean(),4)*100}, ignore_index=True)
            data[feature]=np.where(data[feature].isnull(),"NA","Value")
            data.groupby(feature)[self.dependentVariableName_].median().plot.bar(color=['blue','red'])
            plt.title(feature)
            plt.savefig(Missing_Features_Plot_Dir + feature,dpi = 600)
            plt.close()
        self.missingList_=self.missingList_.sort_values(by='Total Missing Values', ascending=False)
        self.missingList_.to_excel(self.writer_, sheet_name='Summary of Missing values')
        self.missingList_.to_excel(Missing_Features_Dir+"Summary of Missing values.xlsx")
        sns.heatmap(self.dataset_.isnull(),yticklabels=False,cbar=False,cmap="viridis")
        plt.savefig(Missing_Features_Dir + feature,dpi = 600)
        
        plt.close()

    
    def analyzeData(self):
        import pandas as pd
        import os
        import numpy as np 
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        Discrete_Numerical_Features_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Discrete Numerical Features")
        Continous_Numerical_Features_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Continous Numerical Features")
        Continous_Numerical_Features_Histogram_Dir= self.util_.makeDir(Continous_Numerical_Features_Dir,"Continous Numerical Features")
        Continous_Numerical_Features_lognormal_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Continous Numerical lognormal Features")
        Continous_Numerical_Features_Histogram_lognormal_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Continous Numerical lognormal Histograms")
        Continous_Numerical_Features_Boxplot_lognormal_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Continous Numerical lognormal BoxPlot")
        Categorical_Features_Dir= self.util_.makeDir(self.exploratoryDataAnalysisDir_,"Categorical Features")
        self.catagoryList_= pd.DataFrame(columns=['Catagories','Count','Features'])
        self.numericalFeatures_=[feature for feature in self.data_.columns if self.data_[feature].dtype!="O"]
        self.yearFeatures_=[feature for feature in self.data_.columns if "Yr" in feature or "yr" in feature or "Year" in feature or "year" in feature]
        self.numericalFeatures_=list(set(self.numericalFeatures_)-set(self.yearFeatures_))
        self.discreteNumericalFeatures_ =[feature for feature in self.numericalFeatures_ if len(self.data_[feature].unique())<=25]
        self.continousNumericalFeatures_=list(set(self.numericalFeatures_)-set(self.discreteNumericalFeatures_))
        self.catagoricalFeatures_=[feature for feature in self.data_.columns if self.data_[feature].dtype=="O"]
        self.catagoryList_=self.catagoryList_.append({'Catagories':"Year Features" ,'Count':len(self.yearFeatures_),'Features':self.yearFeatures_}, ignore_index=True)
        self.catagoryList_=self.catagoryList_.append({'Catagories':"Discrete Numerical Features",'Count':len(self.discreteNumericalFeatures_),'Features':self.discreteNumericalFeatures_}, ignore_index=True)
        self.catagoryList_=self.catagoryList_.append({'Catagories':"Continous Numerical Features",'Count':len(self.continousNumericalFeatures_),'Features':self.continousNumericalFeatures_}, ignore_index=True)
        self.catagoryList_=self.catagoryList_.append({'Catagories':"Catagorical Features" ,'Count':len(self.catagoricalFeatures_),'Features':self.catagoricalFeatures_  }, ignore_index=True)
        self.catagoryList_.to_excel(self.exploratoryDataAnalysisDir_+"Summary of Features.xlsx")
        self.catagoryList_.to_excel(self.writer_, sheet_name='Summary of Feature Types')
        self.cardinalityList_= pd.DataFrame(columns=['Catagorical Features','Number of Categories'])
    
        for feature in self.discreteNumericalFeatures_:
            data=self.data_.copy()
            data.groupby(feature)[self.dependentVariableName_].median().plot.bar(color=['blue','red','green','yellow','cyan','pink','violet'])
            plt.xlabel(feature)
            plt.ylabel(self.dependentVariableName_)
            plt.title(feature)
            plt.savefig(Discrete_Numerical_Features_Dir + feature,dpi = 600)
            plt.close()

        for feature in self.catagoricalFeatures_:
            self.cardinalityList_=self.cardinalityList_.append({'Catagorical Features': feature,'Number of Categories':len(self.data_[feature].unique())}, ignore_index=True)
            data=self.data_.copy()
            data.groupby(feature)[self.dependentVariableName_].median().plot.bar(color=['blue','red','green','yellow','cyan','pink','violet'])
            plt.xlabel(feature)
            plt.ylabel(self.dependentVariableName_)
            plt.title(feature)
            plt.savefig(Categorical_Features_Dir + feature,dpi = 600)
            plt.close()
        
        for feature in self.continousNumericalFeatures_:
            data=self.data_.copy()
            plt.scatter(data[feature],data[self.dependentVariableName_])
            plt.xlabel(feature)
            plt.ylabel(self.dependentVariableName_)
            plt.title(feature)
            plt.savefig(Continous_Numerical_Features_Dir + feature,dpi = 600)
            plt.close()

        for feature in self.continousNumericalFeatures_:
            data=self.data_.copy()
            data[feature].hist(bins=30)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.title(feature)
            plt.savefig(Continous_Numerical_Features_Histogram_Dir + feature,dpi = 600)
            plt.close()

        for feature in self.continousNumericalFeatures_:
            data=self.data_.copy()
            if 0 in data[feature].unique():
                pass
            else:
                data[feature]=np.log(data[feature])
            plt.scatter(data[feature],data[self.dependentVariableName_])
            plt.xlabel(feature)
            plt.ylabel(self.dependentVariableName_)
            plt.title(feature)
            plt.savefig(Continous_Numerical_Features_lognormal_Dir + feature,dpi = 600)
            plt.close()

        for feature in self.continousNumericalFeatures_:
            data=self.data_.copy()
            if 0 in data[feature].unique():
                pass
            else:
                data[feature]=np.log(data[feature])
            data[feature].hist(bins=30)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.title(feature)
            plt.savefig(Continous_Numerical_Features_Histogram_lognormal_Dir + feature,dpi = 600)
            plt.close()

        for feature in self.continousNumericalFeatures_:
            data=self.data_.copy()
            if 0 in data[feature].unique():
                pass
            else:
                data[feature]=np.log(data[feature])
            data.boxplot(column=feature)
            plt.xlabel(feature)
            plt.ylabel("Data")
            plt.title(feature)
            plt.savefig(Continous_Numerical_Features_Boxplot_lognormal_Dir + feature,dpi = 600)
            plt.close()
    
        self.cardinalityList_.to_excel(Categorical_Features_Dir+"Cardinality of Categorical Features.xlsx")
        self.cardinalityList_.to_excel(self.writer_, sheet_name='Cardinality of Features')
        
        
    def getOutliers_using_Z_Score(self,threshold):
        import pandas as pd
        import numpy as np
        self.outlierThreshold_=threshold
        self.outlierTable_=pd.DataFrame(index=range(len(self.data_)),columns=self.continousNumericalFeatures_)
        for feature in self.continousNumericalFeatures_:
            meanFeature=np.mean(self.data_[feature])
            stdFeature=np.std(self.data_[feature])
            for row in range(len(self.data_[feature])):
                Z_score=(self.data_[feature][row]-meanFeature)/stdFeature
                if abs(Z_score)>threshold:
                    self.outlierTable_[feature][row]=1
                else:
                    self.outlierTable_[feature][row]=0
        self.outlierTable_.to_excel(self.exploratoryDataAnalysisDir_+"Outliers using Z Score.xlsx")       
        self.outlierTable_.to_excel(self.writer_, sheet_name='Outliers using Z Score')
        

    def getOutliers_using_IQR(self):
        import pandas as pd
        import numpy as np
        self.outlierTable_=pd.DataFrame(index=range(len(self.data_)),columns=self.continousNumericalFeatures_)
        for feature in self.continousNumericalFeatures_:
            q1,q3=np.percentile(self.data_[feature],[25,75])
            IQR=q3-q1
            lowerBound= q1-1.5*IQR
            upperBound= q3+1.5*IQR
            for row in range(len(self.data_[feature])):
                
                if self.data_[feature][row]<=lowerBound or self.data_[feature][row]>=upperBound:
                    self.outlierTable_[feature][row]=1
                else:
                    self.outlierTable_[feature][row]=0
        self.outlierTable_.to_excel(self.exploratoryDataAnalysisDir_+"Outliers using IQR.xlsx")       
        self.outlierTable_.to_excel(self.writer_, sheet_name='Outliers using IQR')
        

    def run(self):
        logger.debug("Starting Exploratory Data Analysis")
        logger.debug("Getting Data")
        self.getData()
        logger.debug("Analysing Missing Values")
        self.getMissingValues()
        logger.debug("Classifying Data")
        self.analyzeData()
        logger.debug("Flagging Outliers")
        self.getOutliers_using_Z_Score(3)
        self.writer_.save()
        logger.debug("Exploratory Data Analysis Ends Successfully")
            

class FeatureEngineering:
    def __init__(self,dataFileName,dependentVariableName):
        import pandas as pd
        import os
        import numpy as np
        pd.set_option('display.max_columns', None)
        logger.debug("Starting Featuring Engineering")
        self.EDA_=ExploratoryDataAnalysis(dataFileName,dependentVariableName)
        self.EDA_.run()
        self.util_=Utility()
        self.parentDirectory_ = os.path.dirname(os.getcwd())
        self.featureEngineering_Dir_= self.util_.makeDir(self.parentDirectory_,"Feature Engineering")
        self.FEwriter_ = pd.ExcelWriter(self.featureEngineering_Dir_+'Feature Engineering.xlsx', engine='xlsxwriter')

    def replaceByMedian(self):
        for feature in self.EDA_.missing_Continuous_Numerical_:
            median_val=self.processedData_[feature].median()
            self.processedData_[feature].fillna(median_val,inplace=True)
    def replaceByMean(self):
        for feature in self.EDA_.missing_Continuous_Numerical_:
            mean_val=self.processedData_[feature].mean()
            self.processedData_[feature].fillna(mean_val,inplace=True)
    def replaceByModeDiscreteNumerical(self):
       for feature in self.EDA_.missing_Discrete_Numerical_:
            mode_val=self.processedData_[feature].mode()
            self.processedData_[feature].fillna(mode_val,inplace=True)
     
    def replaceByModeCatagorical(self):
       for feature in self.EDA_.missing_Catagorical_:
            mode_val=self.processedData_[feature].mode()
            self.processedData_[feature].fillna(mode_val,inplace=True)
    def createNewCatagory(self):
        print("Creating New Catagory for Missing Catagorical Data")
        for feature in self.EDA_.missing_Catagorical_:
            self.processedData_[feature]=self.processedData_[feature].fillna("Missing")
            

    def replaceAtRandom(self):
        for feature in self.EDA_.missing_Continuous_Numerical_:
            randomSample=self.processedData_[feature].dropna().sample(self.processedData_[feature].isnull().sum(),random_state=0)
            randomSample.index=self.processedData_[self.processedData_[feature].isnull()].index
            self.processedData_.loc[self.processedData_[feature].isnull(),feature]=randomSample
    def replaceAtRandomDiscreteNumerical(self):
        for feature in self.EDA_.missing_Discrete_Numerical_:
            randomSample=self.processedData_[feature].dropna().sample(self.processedData_[feature].isnull().sum(),random_state=0)
            randomSample.index=self.processedData_[self.processedData_[feature].isnull()].index
            self.processedData_.loc[self.processedData_[feature].isnull(),feature]=randomSample


        
    def replaceMissingValues(self):
        self.processedData_=self.EDA_.data_.copy()
        if len(self.EDA_.missing_Continuous_Numerical_)==0:
            print("There is no missing values of Continious Numerical Features")
        else:
            print("There are {} number of missing values of Continious Numerical Features".format(len(self.EDA_.missing_Continuous_Numerical_)))
            print("Replacing Missing values at Random")
            self.replaceAtRandom()
            
        if len(self.EDA_.missing_Discrete_Numerical_)==0:
            print("There is no missing values of Discrete Numerical Features")
        else:
            print("There are {} number of missing values of Discrete Numerical Features".format(len(self.EDA_.missing_Discrete_Numerical_)))
            self.replaceAtRandomDiscreteNumerical()
        if len(self.EDA_.missing_Catagorical_)==0:
            print("There is no missing values of Categorical Features")
        else:
            print("There are {} number of missing values of Categorical Features".format(len(self.EDA_.missing_Catagorical_)))
            self.createNewCatagory()
    
                
        self.processedData_.to_excel(self.featureEngineering_Dir_+"Missing Value Treated Data.xlsx")
        self.processedData_.to_excel(self.FEwriter_, sheet_name='Missing Value Treated')
        features_With_NA_= [feature for feature in self.processedData_.columns if self.processedData_[feature].isnull().sum()>=1] 
        print("Now Number of Features with Missing Values is : {}".format(len(features_With_NA_)))

    def oneHotEncoding(self,features):
        import pandas as pd
        self.EncodedCatagoricalData_=pd.get_dummies(self.processedData_[features],drop_first=True)
        OtherData=self.processedData_.drop(features,axis=1)
        self.encodedFinalData_=pd.concat((OtherData,self.EncodedCatagoricalData_),axis=1)

    def targetEncoding(self,features,aggregation):
         for feature in features:
             if aggregation=="mean":
                 self.processedData_[feature]=self.processedData_[feature].map(self.processedData_.groupby(feature)[self.EDA_.dependentVariableName_].mean())
             elif aggregation=="median":
                 self.processedData_[feature]=self.processedData_[feature].map(self.processedData_.groupby(feature)[self.EDA_.dependentVariableName_].median())
             elif aggregation=="std":
                 self.processedData_[feature]=self.processedData_[feature].map(self.processedData_.groupby(feature)[self.EDA_.dependentVariableName_].std())
             else:
                 raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median', 'std'".format(aggregation))                
         self.encodedFinalData_=self.processedData_.copy()

    def encodeCatagoricalData(self):
        import pandas as pd
        self.oneHotEncoding(self.EDA_.catagoricalFeatures_)
        #self.targetEncoding(self.EDA_.catagoricalFeatures_,"mean")
        self.encodedFinalData_.to_excel(self.featureEngineering_Dir_+"Missing Value Treated and Encoded Data.xlsx")
        self.encodedFinalData_.to_excel(self.FEwriter_, sheet_name='Encoded Data')
    
    def splitData(self):
        from sklearn.model_selection import  train_test_split
        Y = self.encodedFinalData_[self.EDA_.dependentVariableName_]
        X=self.encodedFinalData_.drop(self.EDA_.dependentVariableName_,axis=1)
        self.X_train_, self.X_test_, self.Y_train_, self.Y_test_ = train_test_split(X, Y, train_size = 0.85, random_state = 21)
        print("Train Data Dimensions : ", self.X_train_.shape)
        print("Test Data Dimensions : ", self.X_test_.shape)
    def scaleVariable(self):
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler()
        dep=[self.EDA_.dependentVariableName_]
        discreteFeatures=list(set(self.EDA_.discreteNumericalFeatures_)-set(dep))
        self.X_train_[self.EDA_.continousNumericalFeatures_]= sc.fit_transform(self.X_train_[self.EDA_.continousNumericalFeatures_])
        self.X_train_[discreteFeatures]= sc.fit_transform(self.X_train_[discreteFeatures])
        self.X_test_[self.EDA_.continousNumericalFeatures_]= sc.fit_transform(self.X_test_[self.EDA_.continousNumericalFeatures_])
        self.X_test_[discreteFeatures]= sc.fit_transform(self.X_test_[discreteFeatures])
        self.X_train_.to_excel(self.featureEngineering_Dir_+"Training Data.xlsx")
        self.X_test_.to_excel(self.featureEngineering_Dir_+"Testing Data.xlsx")

    def trimOutlier_using_Z_Score(self):
        import pandas as pd
        import os
        import numpy as np
        for feature in self.EDA_.continousNumericalFeatures_:
            meanFeature=np.mean(self.EDA_.data_[feature])
            stdFeature=np.std(self.EDA_.data_[feature])
            for row in range(len(self.EDA_.data_[feature])):
                               
                if self.EDA_.outlierTable_[feature][row]==1:
                    if  self.processedData_[feature][row]>=meanFeature:
                        self.processedData_[feature][row]=meanFeature + self.EDA_.outlierThreshold_*stdFeature
                    else:
                        self.processedData_[feature][row]=meanFeature - self.EDA_.outlierThreshold_*stdFeature
                else:
                    pass

        self.processedData_.to_excel(self.featureEngineering_Dir_+"Missing Outlier Treated.xlsx")
        self.processedData_.to_excel(self.FEwriter_, sheet_name='Missing Outlier Treated')
          
class MachineLearning:

    def __init__(self,dataFileName,dependentVariableName):
        import pandas as pd
        import os
        import numpy as np
        
        pd.set_option('display.max_columns', None)
        logger.debug("Starting Classification Module of Machine Learning ")
        self.FE_=FeatureEngineering(dataFileName,dependentVariableName)
        logger.debug("Replacing Missing Values")
        self.FE_.replaceMissingValues()
        logger.debug("Encoding Catagorical Variables")
        self.FE_.encodeCatagoricalData()
        logger.debug("Spliting data into Training and Testing data")
        self.FE_.splitData()
        

        self.util_=Utility()
        self.parentDirectory_ = os.path.dirname(os.getcwd())
        self.Model_Dir_= self.util_.makeDir(self.parentDirectory_,"Machine Learning Models")
    
    
    def NaiveBayesClassifier(self):
        logger.debug("Running NaiveBayesClassifier")
        from sklearn.naive_bayes import MultinomialNB
        print("\n", 'Naive Bayes Classifier')
        self.classifier_ = MultinomialNB(alpha = 1.0)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("NaiveBayesClassifier Ran Successfully")
    def RandomForestClassifier(self):
        logger.debug("Running Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        print("\n", 'Random Forest Classifier')
        self.classifier_ = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=21)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("Random Forest Classifier Ran Successfully")
    def XGBClassifier(self):
        from xgboost import XGBClassifier
        logger.debug("Running XG Boost Classifier")
        print("\n", 'XG Boost Classifier')
        self.classifier_=XGBClassifier(n_estimators=100, max_depth=5,min_child_weight=1, random_state=21, learning_rate=1.0)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("XG Boost Classifier Ran Successfully")
    def AdaBoostClassifier(self):
        logger.debug("Running AdaBoost Classifier")
        from sklearn.ensemble import  AdaBoostClassifier
        print("\n", 'AdaBoost Classifier')
        self.classifier_ = AdaBoostClassifier(n_estimators=200,random_state=21)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("AdaBoost Classifier Ran Successfully")
    def GradientBoostingClassifier(self):
        logger.debug("Running Grdient Boosting Classifier")
        from sklearn.ensemble import  GradientBoostingClassifier
        print("\n", 'Grdient Boosting Classifier')
        self.classifier_ = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=21, learning_rate=1.0)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("Grdient Boosting Classifier Ran Successfully")
    def LinearSupportVectorMachine(self):
        logger.debug("Running Linear Support Vector Machine")
        from sklearn.svm import SVC
        self.FE_.scaleVariable()
        print("\n", 'Linear Support Vector Machine')
        self.classifier_=SVC(kernel="linear",C=1.5,probability=True)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("Linear Support Vector Machine Ran Successfully")
    def KernelSupportVectorMachine(self):
        logger.debug("Running Kernel Support Vector Machine")
        from sklearn.svm import SVC
        self.FE_.scaleVariable()
        print("\n", 'Kernel Support Vector Machine')
        self.classifier_=SVC(kernel="rbf",C=1,gamma='scale',probability=True)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        self.getResult()
        logger.debug("Kernel Support Vector Machine Ran Successfully")
    



        
    def runModel(self):      
        
        self.NaiveBayesClassifier()
        self.RandomForestClassifier()
        self.AdaBoostClassifier()
        self.GradientBoostingClassifier()
        self.XGBClassifier()
        self.LinearSupportVectorMachine()
        self.KernelSupportVectorMachine()
    
        
    
    def getResult(self):
        from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        logger.debug("\n", "Confusion Matrix")
        cm = confusion_matrix(self.FE_.Y_test_, self.Y_pred_)
        logger.debug("\n", cm, "\n")
        #sns.heatmap(cm, square=True, annot=True, cbar=False, fmt = 'g', cmap='RdBu',
                #xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        #plt.xlabel('true label')
        #plt.ylabel('predicted label')
        #plt.show()
        print("\n", "Classification Report", "\n")
        print(classification_report(self.FE_.Y_test_, self.Y_pred_))
        print("Overall Accuracy : ", round(accuracy_score(self.FE_.Y_test_, self.Y_pred_) * 100, 2))
        print("Precision Score : ", round(precision_score(self.FE_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        print("Recall Score : ", round(recall_score(self.FE_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        preds = self.probs_[:,1] # this is the probability for 1, column 0 has probability for 0. Prob(0) + Prob(1) = 1
        fpr, tpr, threshold = roc_curve(self.FE_.Y_test_, preds)
        roc_auc = auc(fpr, tpr)
        print("AUC : ", round(roc_auc * 100, 2), "\n")
       
        plt.figure()
        plt.plot(fpr, tpr, label='Best Model on Test Data (area = %0.2f)' % roc_auc)
        plt.plot([0.0, 1.0], [0, 1],'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RoC-AUC on Test Data')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        #plt.show()
        print("--------------------------------------------------------------------------")
    


                          



       

   
def tuneModel(df):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.model_selection import GridSearchCV    
    Y = df['Label']
    X = df.drop('Label', axis = 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.85, random_state = 21)
    print("Train Data Dimensions : ", X_train.shape)
    print("Test Data Dimensions : ", X_test.shape)
    
    print("**************Tuning XG Boost Classifier*********************")
    grid_params = {'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6],'min_child_weight':[1,2]}
    xgb=XGBClassifier()
    grid_object = GridSearchCV(estimator = xgb, param_grid = grid_params, scoring = 'roc_auc', cv = 10, n_jobs = -1)
    grid_object.fit(X_train, Y_train)
    print("Best Parameters : ", grid_object.best_params_)
    print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
    print("Best model : ", grid_object.best_estimator_)
    Y_pred = grid_object.best_estimator_.predict(X_test)
    probs = grid_object.best_estimator_.predict_proba(X_test)
    getResult(Y_test, Y_pred, probs)
    kfold = KFold(n_splits=10, random_state=25, shuffle=True)
    results = cross_val_score(grid_object.best_estimator_, X_test, Y_test, cv=kfold)
    results = results * 100
    results = np.round(results,2)
    print("Cross Validation Accuracy : ", round(results.mean(), 2))
    print("Cross Validation Accuracy in every fold : ", results)

    print("**************Tuning Grdient Boosting Classifier*********************")
    grid_params = {'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6]}
    gb = GradientBoostingClassifier()
    grid_object = GridSearchCV(estimator = gb, param_grid = grid_params, scoring = 'roc_auc', cv = 10, n_jobs = -1)
    grid_object.fit(X_train, Y_train)
    print("Best Parameters : ", grid_object.best_params_)
    print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
    print("Best model : ", grid_object.best_estimator_)
    Y_pred = grid_object.best_estimator_.predict(X_test)
    probs = grid_object.best_estimator_.predict_proba(X_test)
    getResult(Y_test, Y_pred, probs)
    kfold = KFold(n_splits=10, random_state=25, shuffle=True)
    results = cross_val_score(grid_object.best_estimator_, X_test, Y_test, cv=kfold)
    results = results * 100
    results = np.round(results,2)
    print("Cross Validation Accuracy : ", round(results.mean(), 2))
    print("Cross Validation Accuracy in every fold : ", results)

    print("**************Tuning Ada Boost Classifier*********************")
    grid_params = {'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05]}
    ABC = AdaBoostClassifier()
    grid_object = GridSearchCV(estimator = ABC, param_grid = grid_params, scoring = 'roc_auc', cv = 10, n_jobs = -1)
    grid_object.fit(X_train, Y_train)
    print("Best Parameters : ", grid_object.best_params_)
    print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
    print("Best model : ", grid_object.best_estimator_)
    Y_pred = grid_object.best_estimator_.predict(X_test)
    probs = grid_object.best_estimator_.predict_proba(X_test)
    getResult(Y_test, Y_pred, probs)
    kfold = KFold(n_splits=10, random_state=25, shuffle=True)
    results = cross_val_score(grid_object.best_estimator_, X_test, Y_test, cv=kfold)
    results = results * 100
    results = np.round(results,2)
    print("Cross Validation Accuracy : ", round(results.mean(), 2))
    print("Cross Validation Accuracy in every fold : ", results)
    
    
    print("**************Tuning Random Forest Classifier*********************")
    grid_params = {'n_estimators' : [100,200,300,400,500],'max_depth' : [10, 7, 5, 3],'criterion' : ['entropy', 'gini']}
    RFC = RandomForestClassifier()
    grid_object = GridSearchCV(estimator = RFC, param_grid = grid_params, scoring = 'roc_auc', cv = 10, n_jobs = -1)
    grid_object.fit(X_train, Y_train)
    print("Best Parameters : ", grid_object.best_params_)
    print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
    print("Best model : ", grid_object.best_estimator_)
    Y_pred = grid_object.best_estimator_.predict(X_test)
    probs = grid_object.best_estimator_.predict_proba(X_test)
    getResult(Y_test, Y_pred, probs)
    kfold = KFold(n_splits=10, random_state=25, shuffle=True)
    results = cross_val_score(grid_object.best_estimator_, X_test, Y_test, cv=kfold)
    results = results * 100
    results = np.round(results,2)
    print("Cross Validation Accuracy : ", round(results.mean(), 2))
    print("Cross Validation Accuracy in every fold : ", results)

    print("**************Tuning Naive Bayes Classifier*********************")
    grid_params = {'alpha' : [1,2,3]}
    nb = MultinomialNB()
    grid_object = GridSearchCV(estimator =nb, param_grid = grid_params, scoring = 'roc_auc', cv = 10, n_jobs = -1)
    grid_object.fit(X_train, Y_train)
    print("Best Parameters : ", grid_object.best_params_)
    print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
    print("Best model : ", grid_object.best_estimator_)
    Y_pred = grid_object.best_estimator_.predict(X_test)
    probs = grid_object.best_estimator_.predict_proba(X_test)
    getResult(Y_test, Y_pred, probs)
    kfold = KFold(n_splits=10, random_state=25, shuffle=True)
    results = cross_val_score(grid_object.best_estimator_, X_test, Y_test, cv=kfold)
    results = results * 100
    results = np.round(results,2)
    print("Cross Validation Accuracy : ", round(results.mean(), 2))
    print("Cross Validation Accuracy in every fold : ", results)

    print("**************Tuning Support Vector Machines *********************")
    
    grid_params = [{'kernel': ['rbf'], 'gamma': [1e-2],'C': [ 0.1, 10]},
                   {'kernel': ['sigmoid'], 'gamma': [1e-2],'C': [0.1, 10]},
                   {'kernel': ['linear'], 'C': [0.1, 10]}]
      
    svc=SVC(probability=True)
    grid_object = GridSearchCV(estimator = svc, param_grid = grid_params, scoring = 'roc_auc', cv = 10, n_jobs = -1)
    grid_object.fit(X_train, Y_train)
    print("Best Parameters : ", grid_object.best_params_)
    print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
    print("Best model : ", grid_object.best_estimator_)
    Y_pred = grid_object.best_estimator_.predict(X_test)
    probs = grid_object.best_estimator_.predict_proba(X_test)
    getResult(Y_test, Y_pred, probs)
    kfold = KFold(n_splits=10, random_state=25, shuffle=True)
    results = cross_val_score(grid_object.best_estimator_, X_test, Y_test, cv=kfold)
    results = results * 100
    results = np.round(results,2)
    print("Cross Validation Accuracy : ", round(results.mean(), 2))
    print("Cross Validation Accuracy in every fold : ", results)    





class MongoDB(object):
    def __init__(self,databaseName,collectionName):
        import pymongo
        from pymongo import MongoClient
        import pandas as pd
        import json
        self.databaseName_=databaseName
        self.collectionName_=collectionName
        self.client_= MongoClient("localhost",27017,maxPoolSize=50)
        self.database_=self.client_[self.databaseName_]
        self.collection_=self.database_[self.collectionName_]
    
    def insertData(self,path):
        import pymongo
        from pymongo import MongoClient
        import pandas as pd
        import json
        df=pd.read_csv(path)
        data=df.to_dict("records")
        self.collection_.insert_many(data,ordered=False)
        print("Data Successfully Uploaded in {} collection of Mongo Database".format(self.collectionName_))
    
    def getData(self,databaseName,collectionName):
        import pymongo
        from pymongo import MongoClient
        import pandas as pd
        import json
        db=self.client_[databaseName]
        collection=db[collectionName]
        df = pd.DataFrame(list(collection.find()))
        return df.iloc[:,1:] 

#db=MongoDB("HousingPrice","TrainData")
#db.insertData("train.csv")
#db=MongoDB("HousingPrice","TestData")
#db.insertData("test.csv")
#data=db.getData("HousingPrice","TrainData")




 





        

        
    




       



