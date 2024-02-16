import os
import sys
from src.logger import logging
from src.exception import shippingException


from src.components.ingestion import DataIngestion
from src.components.transformation import DataTransformation
from src.components.training import ModelTrainer
from src.components.evaluation import ModelEvaluation

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)