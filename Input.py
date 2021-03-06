


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
    
    def insertDatafromXL(self,path):
        import pymongo
        from pymongo import MongoClient
        import pandas as pd
        import json
        df=pd.read_csv(path)
        data=df.to_dict("records")
        self.collection_.insert_many(data,ordered=False)
        print("Data Successfully Uploaded in {} collection of Mongo Database".format(self.collectionName_))
    def writeData(self,df,collectionName):
        import pymongo
        from pymongo import MongoClient
        import pandas as pd
        import json
        df=pd.DataFrame(df)
        data=df.to_dict("records")
        db=self.client_[self.databaseName_]
        collection=db[collectionName]
        collection.insert_many(data,ordered=False)
        print(f'Data Successfully Uploaded in table  : {collectionName}  of Mongo Database :  {self.databaseName_}')
    
    def getData(self,databaseName,collectionName):
        import pymongo
        from pymongo import MongoClient
        import pandas as pd
        import json
        db=self.client_[databaseName]
        collection=db[collectionName]
        df = pd.DataFrame(list(collection.find()))
        return df.iloc[:,1:] 



class Input(object):
    def __init__(self,databaseName,collectionName,dependentVariableName):
        self.databaseName_=databaseName
        self.collectionName_=collectionName
        self.dependentVariableName_=dependentVariableName
        self.db_=MongoDB(databaseName,collectionName)
    def readMongoData(self):
        dataset=self.db_.getData(self.databaseName_,self.collectionName_)
        return dataset
    def writeMongoData(self,df,collectionName):
        self.db_.writeData(df,collectionName)
        

class Report(object):
    def __init__(self):
        import pandas as pd
        self.report_ =pd.DataFrame(columns=["Algorithm","Accuracy", "Precision","Recall","AUC"])
        self.predictionReport_ =pd.DataFrame(columns=["   Algorithm   ","Prediction   ","   Probability of 0    ","    Probability of 1   "])
        
    def insertResult(self,list):
        import pandas as pd
        rSeries=pd.Series(list,index=self.report_.columns)
        self.report_=self.report_.append(rSeries,ignore_index=True)
    def insertPredictionResults(self,list):
        import pandas as pd
        rSeries=pd.Series(list,index=self.predictionReport_.columns)
        self.predictionReport_=self.predictionReport_.append(rSeries,ignore_index=True)





