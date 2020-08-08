
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 

from Input import Input ,MongoDB, Report


class Utility:
    
    
    def makeDir(self,parentDirectory, dirName):
        import os
        new_dir = os.path.join(parentDirectory, dirName+'\\')
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        return new_dir
    def SetLogger(self):
        import logging
        logger = logging.getLogger("Classifier")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        file_handler = logging.FileHandler('LogFile.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger
    def stopwatchStart(self):
        import time
        self.start_=time.perf_counter()
    def stopwatchStop(self):
        import time
        self.finish_=time.perf_counter()
    def showTime(self):
        import time
        import numpy as np
        print(f'This operation has finished in {np.round(self.finish_-self.start_,2)}  second(s)')


        


class ExploratoryDataAnalysis:
    
    def __init__(self,input):
        import pandas as pd
        import os
        import numpy as np
        import logging
        self.util_=Utility()
        self.logger_=self.util_.SetLogger()
        #self.dataFileName_=dataFileName
        #self.dependentVariableName_=dependentVariableName
        #self.dataset_= pd.read_csv(dataFileName)
        self.dataset_=input.readMongoData()
        self.dependentVariableName_=input.dependentVariableName_
        self.dataFileName_=input.collectionName_
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
        import logging
        self.logger_.debug("Starting Exploratory Data Analysis")
        self.util_.stopwatchStart()
        self.getData()
        self.logger_.debug("Fetching Data")
        self.getMissingValues()
        self.logger_.info("Catagorizing Data")
        self.analyzeData()
        self.logger_.debug("Geting Outlier using Z score")
        self.getOutliers_using_Z_Score(3)
        self.logger_.debug("Ending Exploratory Data Analysis")
        self.util_.stopwatchStop()
        self.util_.showTime()
        self.writer_.save()
        
            

class FeatureEngineering:
    def __init__(self,input):
        import pandas as pd
        import os
        import numpy as np
        
        pd.set_option('display.max_columns', None)
        self.EDA_=ExploratoryDataAnalysis(input)
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
          
class Classifier:

    def __init__(self,input):
        import pandas as pd
        import os
        import numpy as np
        pd.set_option('display.max_columns', None)
        
        self.input_=input
        self.FE_=FeatureEngineering(input)
        self.FE_.EDA_.logger_.debug("Starting Feature Engineering")
        self.FE_.EDA_.util_.stopwatchStart()
        self.FE_.EDA_.logger_.debug("Replacing Missing Values")
        self.FE_.replaceMissingValues()
        self.FE_.EDA_.logger_.debug("Encoding Catagorical Variables")
        self.FE_.encodeCatagoricalData()
        self.FE_.EDA_.logger_.debug("Spliting Data into Training and Testing ")
        self.FE_.splitData()
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.util_=Utility()
        self.parentDirectory_ = os.path.dirname(os.getcwd())
        self.Model_Dir_= self.util_.makeDir(self.parentDirectory_,"Machine Learning Models")
    
    
    def NaiveBayesClassifier(self):
        
        from sklearn.naive_bayes import MultinomialNB
        print("\n", 'Naive Bayes Classifier')
        self.classifier_ = MultinomialNB(alpha = 1.0)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        return self.getResult('Naive Bayes Classifier')
    
    def tuneNaiveBayesClassifier(self):
        
        import numpy as np
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Naive Bayes Classifier*********************")
        grid_params = {'alpha' : [1,2,3]}
        self.classifier_ = MultinomialNB()
        grid_object = GridSearchCV(estimator =self.classifier_, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_,self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Naive Bayes Classifier')
        
        
    def RandomForestClassifier(self):
       
        from sklearn.ensemble import RandomForestClassifier
        print("\n", 'Random Forest Classifier')
        self.classifier_ = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=21)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
       
        return self.getResult('Random Forest Classifier')
    
    def tuneRandomForestClassifier(self):
        
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        print("**************Tuning Random Forest Classifier*********************")
        grid_params = {'n_estimators' : [100,200,300,400,500],'max_depth' : [10, 7, 5, 3],'criterion' : ['entropy', 'gini']}
        self.classifier_ = RandomForestClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Random Forest Classifier')
        
    def XGBClassifier(self):
        
        from xgboost import XGBClassifier
        print("\n", 'XG Boost Classifier')
        self.classifier_=XGBClassifier(n_estimators=100, max_depth=5,min_child_weight=1, random_state=21, learning_rate=1.0)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
       
        return self.getResult('XG Boost Classifier')
    
    def tuneXGBClassifier(self):
        
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning XG Boost Classifier*********************")
        grid_params = {'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6],'min_child_weight':[1,2]}
        self.classifier_=XGBClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned XG Boost Classifier')
       
    def AdaBoostClassifier(self):
        
        from sklearn.ensemble import  AdaBoostClassifier
        print("\n", 'AdaBoost Classifier')
        self.classifier_ = AdaBoostClassifier(n_estimators=200,random_state=21)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        
        return self.getResult('AdaBoost Classifier')

    def tuneAdaBoostClassifier(self):
        
        import numpy as np
        from sklearn.ensemble import  AdaBoostClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Ada Boost Classifier*********************")
        grid_params = {'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05]}
        self.classifier_ = AdaBoostClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
       
        return self.getResult('Tuned AdaBoost Classifier')
        
    def GradientBoostingClassifier(self):
        
        from sklearn.ensemble import  GradientBoostingClassifier
        print("\n", 'Gradient Boosting Classifier')
        self.classifier_ = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=21, learning_rate=1.0)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
       
        return self.getResult('Gradient Boosting Classifier')

    def tuneGradientBoostingClassifier(self):
       
        import numpy as np
        from sklearn.ensemble import  GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Grdient Boosting Classifier*********************")
        grid_params = {'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6]}
        self.classifier_ = GradientBoostingClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Gradient Boosting Classifier')
        
    def LinearSupportVectorMachine(self):
        from sklearn.svm import SVC
        self.FE_.scaleVariable()
        print("\n", 'Linear Support Vector Machine')
        self.classifier_=SVC(kernel="linear",C=1.5,probability=True)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        return self.getResult('Linear Support Vector Machine')
       
    def KernelSupportVectorMachine(self):
        
        from sklearn.svm import SVC
        self.FE_.scaleVariable()
        print("\n", 'Kernel Support Vector Machine')
        self.classifier_=SVC(kernel="rbf",C=1,gamma='scale',probability=True)
        self.classifier_.fit(self.FE_.X_train_, self.FE_.Y_train_)
        self.Y_pred_ = self.classifier_.predict(self.FE_.X_test_)
        self.probs_ = self.classifier_.predict_proba(self.FE_.X_test_)
        
        return self.getResult('Kernel Support Vector Machine')
    
    def tuneKernelSupportVectorMachine(self):
        
        import numpy as np
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        self.FE_.scaleVariable()
        print("**************Tuning Kernel Support Vector Machine*********************")
        grid_params = [{'kernel': ['rbf'], 'gamma': [1e-2]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-2]},
                         {'kernel': ['linear']}]
      
        self.classifier_=SVC(probability=True)
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Support Vector Machine')
       
           
    def runModel(self):
        self.report_=Report()
        self.FE_.EDA_.logger_.debug("Running Naive Bayes Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.NaiveBayesClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Running Random Forest Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.RandomForestClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Running AdaBoost Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.AdaBoostClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Running Gradient Boosting Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.GradientBoostingClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Running XGBClassifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.XGBClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Running Kernel Support Vector Machine ")
        self.FE_.EDA_.util_.stopwatchStart()
        #self.LinearSupportVectorMachine()
        lst=self.KernelSupportVectorMachine()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.report_.report_= self.report_.report_.sort_values(["Accuracy"], ascending =False)
        print(self.report_.report_)
        self.input_.writeMongoData(self.report_.report_,"ModelComparisonReport")
        self.FE_.EDA_.logger_.debug("Ending Program. Thanks for your visit ")
    
    def compareModel(self):
        self.report_=Report()
        self.FE_.EDA_.logger_.debug("Tuning Naive Bayes Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneNaiveBayesClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Tuning Random Forest Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneRandomForestClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Tuning AdaBoost Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneAdaBoostClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Tuning Gradient Boosting Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneGradientBoostingClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Tuning XGBClassifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneXGBClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Tuning  Support Vector Machine ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneKernelSupportVectorMachine()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.report_.report_= self.report_.report_.sort_values(["Accuracy"], ascending =False)
        print(self.report_.report_)
        self.input_.writeMongoData(self.report_.report_,"TunedModelComparisonReport")
        self.FE_.EDA_.logger_.debug("Ending Program. Thanks for your visit ")

    def compareModel1(self):
        self.algoCall_={"Tuning Naive Bayes Classifier ":self.tuneNaiveBayesClassifier(),
                        "Tuning Random Forest Classifier":self.tuneRandomForestClassifier(),
                        "Tuning AdaBoost Classifier":self.tuneAdaBoostClassifier(),
                        "Tuning Gradient Boosting Classifier":self.tuneGradientBoostingClassifier(),
                        "Tuning XGBClassifier":self.tuneXGBClassifier(),
                        "Tuning Support Vector Machine":self.tuneKernelSupportVectorMachine()}
        self.report_=Report()               
        for key in self.algoCall_:
             print("Start Stopwatch")
             self.FE_.EDA_.util_.stopwatchStart()
             print("Stopwatch Started ")
             self.report_.insertResult(self.algoCall_[key])
             print("Stop Stopwatch")
             self.FE_.EDA_.util_.stopwatchStop()
             print("Stopwatch Stopped ")
             self.FE_.EDA_.util_.showTime()
             print("Time Displayed ")
             
        self.report_.report_= self.report_.report_.sort_values(["Accuracy"], ascending =False)
        print(self.report_.report_)
        self.FE_.EDA_.logger_.debug("Ending Program. Thanks for your visit ")
    
    def getResult(self,algoName):
        from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        report=[algoName]
        print("\n", "Confusion Matrix")
        cm = confusion_matrix(self.FE_.Y_test_, self.Y_pred_)
        print("\n", cm, "\n")
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
        report.append(round(accuracy_score(self.FE_.Y_test_, self.Y_pred_) * 100, 2))
        report.append(round(precision_score(self.FE_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        report.append(round(recall_score(self.FE_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        report.append(round(roc_auc * 100, 2))

        plt.figure()
        plt.plot(fpr, tpr, label='Best Model on Test Data (area = %0.2f)' % roc_auc)
        plt.plot([0.0, 1.0], [0, 1],'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RoC-AUC on Test Data')
        plt.legend(loc="lower right")
        #plt.savefig('Log_ROC')
        #plt.show()
        return report
    













 





        

        
    




       



