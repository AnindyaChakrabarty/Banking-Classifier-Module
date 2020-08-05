
import Classifier


class App:

    def __init__(self,dataFileName,dependentVariableName):
        self.logging()
        self.__Classifier=MachineLearning(dataFileName,dependentVariableName)

    def logging(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        file_handler = logging.FileHandler('MachineLearning.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    def runClassifier(self):
        self.__Classifier.runModel()



model= App("Data.csv","Exited")
model.runClassifier()

   





