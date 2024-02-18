import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import shippingException
from src.utils.utils import load_object


class PredictPipeline:

    
    def __init__(self):
        pass

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

    def __init__ (self, unit_of_measure_per_pack:float,
                  line_item_quantity:float,
                  pack_price:float,
                  unit_price:float, 
                  freight_cost_usd:float,
                  line_item_insurance_usd:float,
                  fulfill_via:str,
                  vendor_inco_term:str,
                  shipment_mode:str,
                  first_line_designation:str):
        
        self.unit_of_measure_per_pack=unit_of_measure_per_pack
        self.line_item_quantity=line_item_quantity
        self.pack_price=pack_price
        self.unit_price=unit_price
        self.freight_cost_usd=freight_cost_usd
        self.line_item_insurance_usd = line_item_insurance_usd
        self.fulfill_via=fulfill_via
        self.vendor_inco_term=vendor_inco_term
        self.shipment_mode=shipment_mode
        self.first_line_designation=first_line_designation

    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'unit_of_measure_(per_pack)':[self.unit_of_measure_per_pack],
                    'line_item_quantity':[self.line_item_quantity],
                    'pack_price':[self.pack_price],
                    'unit_price':[self.unit_price],
                    'freight_cost_(usd)':[self.freight_cost_usd],
                    'line_item_insurance_(usd)':[self.line_item_insurance_usd],
                    'fulfill_via':[self.fulfill_via],
                    'vendor_inco_term':[self.vendor_inco_term],
                    'shipment_mode':[self.shipment_mode],
                    'first_line_designation':[self.first_line_designation]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise shippingException(e,sys)
            



    

    
    



