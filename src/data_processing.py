import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.processed_data_path = PROCESSED_DATA_PATH
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info("Read data successfully")
        except Exception as e:
            logger.error(f"Error while reading data {e}")
            raise CustomException("Failed to read data", e)
            
    def handle_outliers(self, column):
        try:
            logger.info("Starting to handle outliers..")
                
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)

            IQR = Q3 - Q1
                # oulier detection
            lower_value = Q1 - 1.5 * IQR 
            upper_value = Q3 + 1.5 * IQR

            sepal_median = np.median(self.df[column])

            for i in self.df[column]:
                if i > upper_value or i < lower_value:
                    self.df[column] = self.df[column].replace(i,sepal_median)
            logger.info("Handled outlier successfully...")

        except Exception as e:
            logger.error(f"Error while handling outliers {e}") 
            raise CustomException("Faailed to handle outliers",e)               

    def split_data(self):
        try:
            X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            y = self.df['Species']

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
            logger.info("splitted data successfully..")

            joblib.dump(X_train, X_TRAIN_PROCESSED)
            joblib.dump(X_test, X_TEST_PROCESSED)
            joblib.dump(y_train, Y_TRAIN_PROCESSED)
            joblib.dump(y_test, Y_TEST_PROCESSED)

            logger.info("Files saved successfully for data processing steps..")
        except Exception as e:
            logger.info(f"Error while saving files {e}")
            raise CustomException("Failed to save files")
        
    def run(self):
        self.load_data()
        self.handle_outliers("SepalWidthCm")
        self.split_data()

if __name__ == "__main__":
    data_processor = DataProcessing(FILE_PATH)
    data_processor.run()