import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import shippingException
import os
import sys
from dataclasses import dataclass
from pathlib import Path


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()



    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            numerical_cols=['Unit of Measure (Per Pack)','Line Item Quantity','Pack Price',
                            'Unit Price',
                            'Line Item Insurance (USD)','Freight Cost (USD)']
            
            categorical_cols=['Fulfill Via','Vendor INCO Term','Shipment Mode','First Line Designation']

            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('scaler',StandardScaler())
                ])
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder())
                ])
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols),
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise shippingException(e,sys)
    @staticmethod
    def _outlier_capping(df,col):
        """
        Method Name :   _outlier_capping

        Description :   This method performs outlier capping in the dataframe. 

        Output      :   DataFrame. 
        """
        logging.info("Entered _outlier_capping method of Data_Transformation class")
        try:
            logging.info("Performing _outlier_capping for columns in the dataframe")
            percentile25 = df[col].quantile(0.25)  # calculating 25 percentile
            percentile75 = df[col].quantile(0.75)  # calculating 75 percentile

            # Calculating upper limit and lower limit
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr

            # Capping the outliers
            df.loc[(df[col] > upper_limit), col] = upper_limit
            df.loc[(df[col] < lower_limit), col] = lower_limit
            logging.info(
                "Performed _outlier_capping method of Data_Transformation class"
            )

            logging.info("Exited _outlier_capping method of Data_Transformation class")
            return df

        except Exception as e:
            raise shippingException(e, sys) from e
    ## To removing the irregularity in the freight cost data
    @staticmethod
    def _trans_freight_cost(x):
                if x.find("See")!=-1:
                    return np.nan
                elif x=="Freight Included in Commodity Cost" or x=="Invoiced Separately":
                    return 0
                else:
                    return x
            
    
    def initialize_data_transformation(self,train_path,test_path):

        try:
            self.train_df=pd.read_csv(train_path)
            self.test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{self.train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{self.test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()

            target_column_name='Line Item Value'

            drop_columns = [target_column_name,'ID', 'Project Code', 'PQ ', 'PO / SO ', 'ASN/DN ', 'Country',
                            'Managed By','PQ First Sent to Client Date', 'PO Sent to Vendor Date','Scheduled Delivery Date', 
                            'Delivered to Client Date','Delivery Recorded Date', 'Product Group', 'Sub Classification',
                            'Vendor', 'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage','Dosage Form',
                            'Manufacturing Site','Weight (Kilograms)'
                             ]
            
            
            self.train_df["Freight Cost (USD)"] = self.train_df["Freight Cost (USD)"].apply(DataTransformation._trans_freight_cost)

            self.test_df["Freight Cost (USD)"] = self.test_df["Freight Cost (USD)"].apply(DataTransformation._trans_freight_cost)
           
            numerical_cols=['Unit of Measure (Per Pack)','Line Item Quantity','Line Item Value','Pack Price',
                            'Unit Price',
                            'Line Item Insurance (USD)','Freight Cost (USD)']
            
            for col in numerical_cols:
                self.train_df[col] = self.train_df[col].fillna(self.train_df[col].median())
                self.test_df[col] = self.test_df[col].fillna(self.test_df[col].median())
                
            self.train_df["Freight Cost (USD)"]=self.train_df["Freight Cost (USD)"].astype("float")
            self.test_df["Freight Cost (USD)"]=self.test_df["Freight Cost (USD)"].astype("float")
                
            
                
            logging.info("NaN values are being filled") 

            # Outlier capping
            logging.info("Got a list of numerical_col")
            [self._outlier_capping(self.train_df,col) for col in numerical_cols]
            logging.info("Outlier capped in train df")
            [self._outlier_capping(self.test_df,col) for col in numerical_cols]
            logging.info("Outlier capped in test df")

            input_feature_train_df = self.train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=self.train_df[target_column_name]
            
            input_feature_test_df=self.test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=self.test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise shippingException(e,sys)
    


            


            


            
                            
            
