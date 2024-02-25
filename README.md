
# Consignment Price Prediction

The Consignment Pricing Prediction project focuses on developing an accurate predictive model 
for consignment pricing within the shipping industry. The objective is to forecast the line item 
value, considering various factors influencing consignment prices.


## Badges


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


## About
The market for logistics analytics is expected to develop at a CAGR of 17.3 percent from 2019 to 
2024, more than doubling in size. This data demonstrates how logistics organizations understand 
the advantages of being able to predict what will happen in the future with a decent degree of 
certainty. Logistics leaders may use this data to address supply chain difficulties, cut costs, and 
enhance service levels all at the same time. The main goal is to predict the consignment pricing 
based on the available factors in the dataset.

## Data 
Data for the project taken fron the Kaggle https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data.

## Architecture

![Untitled Diagram drawio](https://github.com/chittettu/consignment/assets/105189151/868d5bb4-c81d-4cdc-a18b-7423132f580c)

## Model information

           Model Name              R2 score 
        1. Linear Regression         94.67        
        2. Elasticnet                86.82
        3. Random forest             97.82
        4. Xgboost                   97.4
        
## Result and analysis
After model training, the model shows a R-squared value of 0.9782 (97.82% accuracy) on the test data, signifying a robust level of predictive accuracy.
## Installation
To run the code, first clone this repository and navigate to the project directory:

           git clone https://github.com/chittettu/consignment.git

Create a virtual environment

           conda create -p env python==3.8 -y
           conda activate env/
           
To run this project, you will need Python packages present in the requirements file

            pip install -r requirements.txt

Then, run the app.py file to start the Flask web application:

            streamlit run app.py

## Web deployment

Deployed in:
* AWS
* Streamlit cloud

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Run the project

* Clone the project
* pip install -r requirements.txt
* streamlit run app.py (for running in local host)


