import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import shippingException
from src.utils.utils import load_object


class PredictPipeline:

    
    def __init__(self):
        print("init.. the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_fea=preprocessor.transform(features)
            pred=model.predict(scaled_fea)

            return pred

        except Exception as e:
            raise shippingException(e,sys)
class CustomData:
    def __init__(self,
                 Unit of Measure (Per Pack):int,
                 Line Item Quantity:int,
                 Pack Price:float,
                 :float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
