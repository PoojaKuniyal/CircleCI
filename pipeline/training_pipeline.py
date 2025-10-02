# will run this inside docker to train model

from config.paths_config import *
from src.data_processing import DataProcessing
from src.model_training import ModelTraining

if __name__ =="__main__":
    data_processor = DataProcessing(FILE_PATH)
    data_processor.run()

    trainer = ModelTraining()
    trainer.run()
    