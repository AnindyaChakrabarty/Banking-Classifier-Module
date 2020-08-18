
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 

from Input import Input ,MongoDB, Report


class Utility:
    
        
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
       
        import numpy as np
        import logging
        self.util_=Utility()
        self.logger_=self.util_.SetLogger()
        self.dataset_=input.readMongoData()
        self.dependentVariableName_=input.dependentVariableName_
        self.dataFileName_=input.collectionName_
        
    
                
    def getY(self):
        self.Y_=self.dataset_[self.dependentVariableName_]
    def getX(self):
        Header=list(self.dataset_.columns)
        Header.remove(self.dependentVariableName_)
        self.X_=self.dataset_[Header]
    def getData(self):
        self.Header_=list(self.dataset_.columns)
        self.Header_.remove('RowNumber')
        self.Header_.remove('CustomerId') 
        self.data_=self.dataset_[self.Header_]
    def isImbalence(self,threshold):
        imbl=self.data_[self.dependentVariableName_].value_counts()
        if (imbl[1]/imbl[0]<threshold or imbl[0]/imbl[1]<threshold):
            print(f'We have imbalence dataset with count of 1 in Total Data : {imbl[1]} and count of 0 in Total Data : {imbl[0]}')
        else:
            print(f'We do not have imbalence dataset with count of 1 in Total Data : {imbl[1]} and count of 0 in Total Data : {imbl[0]}')
        return (imbl[1]/imbl[0]<threshold or imbl[0]/imbl[1]<threshold)
        
    def getMissingValues(self):
        import pandas as pd
        import numpy as np            
        data=self.data_.copy()
        self.features_With_NA_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1]
        self.missing_Continuous_Numerical_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1 and self.data_[feature].dtype!="O" and feature !="Id" and len(self.data_[feature].unique())>25]
        self.missing_Discrete_Numerical_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1 and self.data_[feature].dtype!="O" and feature !="Id" and len(self.data_[feature].unique())<=25]
        self.missing_Catagorical_= [feature for feature in self.data_.columns if self.data_[feature].isnull().sum()>=1 and self.data_[feature].dtype=="O"]
        self.missingList_= pd.DataFrame(columns=['Features','Total Missing Values','% Missing Values'])
        for feature in self.features_With_NA_:
            self.missingList_ = self.missingList_.append({'Features': feature,'Total Missing Values': np.round(self.data_[feature].isnull().sum(),0), '% Missing Values': np.round(self.data_[feature].isnull().mean(),4)*100}, ignore_index=True)
        self.missingList_=self.missingList_.sort_values(by='Total Missing Values', ascending=False)
        

    
    def analyzeData(self):
        import pandas as pd
        import numpy as np   
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
        self.cardinalityList_= pd.DataFrame(columns=['Catagorical Features','Number of Categories'])
      
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
        
        

    def run(self):
        import logging
        self.logger_.debug("***********************************************************")
        self.logger_.debug(" Anindya Chakrabarty Welcomes you to Classification Module ")
        self.logger_.debug("***********************************************************")
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
        
        
            

class FeatureEngineering:
    def __init__(self,input):
        import pandas as pd
        import numpy as np
        self.EDA_=ExploratoryDataAnalysis(input)
        self.EDA_.run()
        self.util_=Utility()
       

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
        
       
   
    def splitData(self):
        from sklearn.model_selection import  train_test_split
        from collections import Counter
        Y = self.encodedFinalData_[self.EDA_.dependentVariableName_]
        X=self.encodedFinalData_.drop(self.EDA_.dependentVariableName_,axis=1)
        self.X_train_, self.X_test_, self.Y_train_, self.Y_test_ = train_test_split(X, Y, train_size = 0.85, random_state = 21)
        print('Original  Training Dataset Shape {}'.format(Counter(self.Y_train_)))
        print('Original  Testing Dataset Shape {}'.format(Counter(self.Y_test_)))
    def overSampling(self,ratio):
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter
        os=RandomOverSampler(ratio)
        self.X_train_, self.Y_train_ =os.fit_resample(self.X_train_, self.Y_train_)
        print('Over Sampled  Training Dataset Shape {}'.format(Counter(self.Y_train_)))
    def SMOTE(self,k):
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        smote=SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=100)
        self.X_train_, self.Y_train_ =smote.fit_resample(self.X_train_, self.Y_train_)
        print('SMOTE  Training Dataset Shape {}'.format(Counter(self.Y_train_)))

    def handlingImbalanceData(self):
        if (self.EDA_.isImbalence(0.5)):
            #self.overSampling(1)
            self.SMOTE(1)
        else:
            print("Data set is balanced and hence no changes made")
         
        
    def scaleVariable(self):
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler()
        dep=[self.EDA_.dependentVariableName_]
        discreteFeatures=list(set(self.EDA_.discreteNumericalFeatures_)-set(dep))
        self.X_train_[self.EDA_.continousNumericalFeatures_]= sc.fit_transform(self.X_train_[self.EDA_.continousNumericalFeatures_])
        self.X_train_[discreteFeatures]= sc.fit_transform(self.X_train_[discreteFeatures])
        self.X_test_[self.EDA_.continousNumericalFeatures_]= sc.fit_transform(self.X_test_[self.EDA_.continousNumericalFeatures_])
        self.X_test_[discreteFeatures]= sc.fit_transform(self.X_test_[discreteFeatures])
        

    def trimOutlier_using_Z_Score(self):
        import pandas as pd
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

        
    
    def run(self):
        self.EDA_.logger_.debug(" ***************************************************************")
        self.EDA_.logger_.debug(" Anindya Chakrabarty Welcomes you to Fearure Engineering Module ")
        self.EDA_.logger_.debug(" ***************************************************************")
        self.EDA_.logger_.debug("Starting Feature Engineering")
        self.EDA_.util_.stopwatchStart()
        self.EDA_.logger_.debug("Replacing Missing Values")
        self.replaceMissingValues()
        self.EDA_.logger_.debug("Encoding Catagorical Variables")
        self.encodeCatagoricalData()
        self.EDA_.logger_.debug("Spliting Data into Training and Testing ")
        self.splitData()
        self.EDA_.logger_.debug("Handling Imbalance Dataset ")
        self.handlingImbalanceData()
        self.EDA_.util_.stopwatchStop()
        self.EDA_.util_.showTime()

        
          
class Classifier:

    def __init__(self,input):
        import pandas as pd
        import numpy as np
        self.input_=input
        self.bestModels_={}
        self.FE_=FeatureEngineering(input)
        self.FE_.run()
        self.util_=Utility()
        
    
    def getHyperParameters(self):
         
         self.grid_params_NaiveBayesClassifier_ = {'alpha' : [1,2,3]}
         self.grid_params_RandomForestClassifier_ = {'n_estimators' : [100,200,300,400,500],'max_depth' : [10, 7, 5, 3],'criterion' : ['entropy', 'gini']}
         self.grid_params_XGBClassifier_={'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6],'min_child_weight':[1,2]}
         self.grid_params_AdaBoostClassifier_={'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05]}
         self.grid_params_GradientBoostingClassifier_={'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6]}
         self.grid_params_KernelSupportVectorMachine_=[{'kernel': ['rbf','sigmoid','linear'], 'gamma': [1e-2]}]
         self.grid_params_LogisticRegression_= {'C' : [0.0001, 0.01, 0.05, 0.2, 1],'penalty' : ['l1', 'l2']} 
         self.grid_params_ExtraTreesClassifier_={'n_estimators' : [100,200,300,400,500],'max_depth' : [10, 7, 5, 3],'criterion' : ['entropy', 'gini']}
            
    def tuneNaiveBayesClassifier(self):
        
        import numpy as np
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Naive Bayes Classifier*********************")
        self.classifier_ = MultinomialNB()
        grid_object = GridSearchCV(estimator =self.classifier_, param_grid = self.grid_params_NaiveBayesClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_,self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Naive Bayes Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Naive Bayes Classifier')
   
    def tuneRandomForestClassifier(self):
        
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        print("**************Tuning Random Forest Classifier*********************")
        
        self.classifier_ = RandomForestClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_RandomForestClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Random Forest Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Random Forest Classifier')
        
    def tuneXGBClassifier(self):
        
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning XG Boost Classifier*********************")
        
        self.classifier_=XGBClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_XGBClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'XG Boost Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned XG Boost Classifier')
    
    def tuneAdaBoostClassifier(self):
        
        import numpy as np
        from sklearn.ensemble import  AdaBoostClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Ada Boost Classifier*********************")
       
        self.classifier_ = AdaBoostClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_AdaBoostClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Ada Boost Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
       
        return self.getResult('Tuned AdaBoost Classifier')
   
    def tuneGradientBoostingClassifier(self):
       
        import numpy as np
        from sklearn.ensemble import  GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Grdient Boosting Classifier*********************")
       
        self.classifier_ = GradientBoostingClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_GradientBoostingClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Grdient Boosting Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Gradient Boosting Classifier')
        
    def tuneKernelSupportVectorMachine(self):
        
        import numpy as np
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        self.FE_.scaleVariable()
        print("**************Tuning Kernel Support Vector Machine*********************")
           
        self.classifier_=SVC(probability=True)
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_KernelSupportVectorMachine_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Kernel Support Vector Machine':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Support Vector Machine')
    
    def tuneLogisticRegression(self):
        
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        self.FE_.scaleVariable()
        print("**************Tuning Logistic Regression*********************")
        
        self.classifier_=LogisticRegression()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_LogisticRegression_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Logistic Regression':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Logistic Regression')

    def tuneExtraTreesClassifier(self):
        
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        from sklearn.ensemble import ExtraTreesClassifier
        print("**************Tuning Extra Trees Classifier*********************")
        
        self.classifier_ = ExtraTreesClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_ExtraTreesClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.FE_.X_train_, self.FE_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Extra Trees Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.FE_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.FE_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.FE_.X_test_, self.FE_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Extra Trees Classifier')      
        
    def compareModel(self):
        import pickle
        self.report_=Report()
        self.getHyperParameters()
        self.FE_.EDA_.logger_.debug("Tuning Logistic Regression ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneLogisticRegression()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
        self.FE_.EDA_.logger_.debug("Tuning  Extra Trees Classifier ")
        self.FE_.EDA_.util_.stopwatchStart()
        lst=self.tuneExtraTreesClassifier()
        self.report_.insertResult(lst)
        self.FE_.EDA_.util_.stopwatchStop()
        self.FE_.EDA_.util_.showTime()
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
        self.FE_.EDA_.logger_.debug("Saving Final Models in Pickle")
        pickle.dump(self.bestModels_,open('model.pkl','wb'))
        self.FE_.EDA_.logger_.debug("Ending Model Calibration ")

    def compareModel1(self):
        self.getHyperParameters()
        self.algoCall_={"Tuning Logistic Regression ":self.tuneLogisticRegression(),
                        "Tuning Naive Bayes Classifier ":self.tuneNaiveBayesClassifier(),
                        "Tuning Random Forest Classifier":self.tuneRandomForestClassifier(),
                        "Tuning  Extra Trees Classifier ":self.tuneExtraTreesClassifier(),
                        "Tuning AdaBoost Classifier":self.tuneAdaBoostClassifier(),
                        "Tuning Gradient Boosting Classifier":self.tuneGradientBoostingClassifier(),
                        "Tuning XGBClassifier":self.tuneXGBClassifier(),
                        "Tuning Support Vector Machine":self.tuneKernelSupportVectorMachine()}
        self.report_=Report()               
        for key in self.algoCall_:
            self.report_.insertResult(self.algoCall_[key])    
        self.report_.report_= self.report_.report_.sort_values(["Accuracy"], ascending =False)
        
    
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

    def predict(self,newData):
        self.FE_.EDA_.logger_.debug("Starting Model prediction ")
        import pandas as pd
        import numpy as np
        self.predictionReport_=Report()
        self.newInput_=Input(self.input_.databaseName_,newData,self.input_.dependentVariableName_)
        self.newDataSet_=self.newInput_.readMongoData()
        self.FE_.EDA_.Header_.remove(self.input_.dependentVariableName_)     
        self.newData_=self.newDataSet_[self.FE_.EDA_.Header_].copy()
        self.newData_ = pd.get_dummies(self.newData_,drop_first=False)
        self.newData_=self.newData_.reindex(columns=list(self.FE_.X_train_.columns),fill_value=0)
        
        for key in self.bestModels_:
            self.predictionReport_.insertPredictionResults([key,int(self.bestModels_[key].predict(self.newData_)),int(np.round(self.bestModels_[key].predict_proba(self.newData_)[0][0],2)*100),int(np.round(self.bestModels_[key].predict_proba(self.newData_)[0][1],2)*100)])              
        print(self.predictionReport_.predictionReport_)
        self.FE_.EDA_.logger_.debug("Ending Model prediction. Good Bye")

    

                              
            
            
        
   
        

    
  












 





        

        
    




       



