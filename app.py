import streamlit as st
import pandas as pd
from src.pipeline.prediction import CustomData, PredictPipeline

# Load the preprocessor and model
predictor = PredictPipeline()


# Streamlit app
def main():
    
    st.title('Consignment Prediction web app')

    # Collect user input
    unit_of_measure_per_pack = st.number_input('Unit of Measure per Pack', value=0.0)
    line_item_quantity = st.number_input('Line Item Quantity', value=0.0)
    pack_price = st.number_input('Pack Price', value=0.0)
    unit_price = st.number_input('Unit Price', value=0.0)
    freight_cost_usd = st.number_input('Freight Cost (USD)', value=0.0)
    line_item_insurance_usd = st.number_input('Line Item Insurance (USD)', value=0.0)


    fulfill_via = st.selectbox('Fulfill Via', ['From RDC', 'Direct Drop'])
    vendor_inco_term = st.selectbox('Vendor Inco Term', ['N/A - From RDC', 'EXW', 'DDP','FCA','CIP','DDU','DAP','CIF'])
    shipment_mode = st.selectbox('Shipment Mode', ['Air', 'Truck', 'Air Charter','Ocean'])
    first_line_designation = st.selectbox('First Line Designation', ['Yes', 'No'])

    # Create CustomData instance
    custom_data = CustomData(
        unit_of_measure_per_pack, line_item_quantity, pack_price, unit_price,
        freight_cost_usd, line_item_insurance_usd, fulfill_via,
        vendor_inco_term, shipment_mode, first_line_designation
    )

    # Convert CustomData to DataFrame
    df = custom_data.get_data_as_dataframe()

    # Make predictions
    if st.button('Predict'):
        prediction = predictor.predict(df)
        st.success(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()





