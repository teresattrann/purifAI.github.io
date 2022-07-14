from operator import index
import dash
import purifai
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import base64
import datetime
import dash_bootstrap_components as dbc     
from purifai.methodSelection import model_selection
import wget
import os


names = ['MolWt', 'exactMolWt', 'qed', 'TPSA', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'FractionCSP3', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'FpDensityMorgan1', 'BalabanJ', 'BertzCT', 'HallKierAlpha', 'Ipc', 'Kappa2', 'LabuteASA', 'PEOE_VSA10', 'PEOE_VSA2', 'SMR_VSA10', 'SMR_VSA4', 'SlogP_VSA2', 'SlogP_VSA6','MaxEStateIndex', 'MinEStateIndex', 'EState_VSA3', 'EState_VSA8', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount']

# model_predictor = model_selection("static/models/spe_brf_model.pkl",
#                             "static/models/spe_scaler.pkl",
#                             "static/models/lcms_brf_model.pkl",
#                             "static/models/lcms_scaler.pkl")

cwd = os.getcwd()
    
url = 'https://github.com/jenamis/purifAI/raw/main/machine_learning/SPE/models/'
if not os.path.exists(os.getcwd() + '/spe_xgb_model.pkl'):
    wget.download(url+ 'spe_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/spe_scaler.pkl'):
    wget.download(url+ 'spe_scaler.pkl')
    
url= 'https://github.com/jenamis/purifAI/raw/main/machine_learning/LCMS/models/'
if not os.path.exists(os.getcwd() + '/lcms_xgb_model.pkl'):
    wget.download(url+ 'lcms_xgb_model.pkl')
if not os.path.exists(os.getcwd() + '/lcms_scaler.pkl'):
    wget.download(url+ 'lcms_scaler.pkl')
    
spe_xgb_model = cwd + '/spe_xgb_model.pkl'
spe_scaler = cwd + '/spe_scaler.pkl'
lcms_xgb_model = cwd + '/lcms_xgb_model.pkl'
lcms_scaler = cwd + '/lcms_scaler.pkl'

model_predictor = model_selection(spe_xgb_model, 
                            spe_scaler,
                            lcms_xgb_model,
                            lcms_scaler)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Textarea(
        id='textarea-input',
        style={'width': '100%', 'height': 50 },
    ),
    # html.Div(id='output-data', style={'whiteSpace': 'pre-line'}),
    # html.Div(dcc.Input(id='input-data', type='text')),
    dbc.Button('Predict', id='submit-button', className="me-2", size="lrg", n_clicks=0),
    html.Div(id='submit-text'),
    html.Div(id='textarea-output', style={'whiteSpace': 'pre-line'}),
    html.Div(id='table-output')
]
, style={'margin-bottom': '10px',
              'textAlign':'center',
              'width': '1200px',
              'margin':'auto'}
)

@app.callback(
    Output("table-output", "children"), [Input("submit-button", "n_clicks"), Input("textarea-input", "value")],
)
def on_button_click(n_clicks, contents):
    if n_clicks > 0:
        table = html.Div()
        columns = ['SMILES']
        prediction_df = pd.DataFrame(columns=columns)
        descriptors_df = pd.DataFrame(columns=names)
        df = pd.DataFrame(columns=[columns + names])
            
        
        print(contents)
        smiles_list = contents.split('\n')
        # smiles_list.pop()
        prediction_df['SMILES'] = smiles_list
        print(smiles_list)
        
        x = 0
        for i in range(len(prediction_df)):
            smiles = prediction_df.loc[i, 'SMILES']
            print(f"SMILES entry received: {i}")

            descriptors = model_selection.calculate_descriptors(self=model_selection, smiles=smiles)
            descriptors_temp = pd.DataFrame([descriptors],columns=names)
            descriptors_df = pd.concat([descriptors_df, descriptors_temp])
            print(f"Molecular descriptors calculated for entry {i}:")
            print(descriptors_df.head())
                            
            predicted_SPE_method = model_predictor.RunSPEPrediction(smiles)
            print(f"Test 1 {predicted_SPE_method}")
            prediction_df.loc[i, "Predicted SPE Method"] = str(predicted_SPE_method)
            print(f"SPE prediction succesful...\n Predicted {predicted_SPE_method} SPE method for entry {i}")
            
            predicted_LCMS_method = model_predictor.RunLCMSPrediction(smiles)
            print(f"Test 2 {predicted_LCMS_method}")
            prediction_df.loc[i, "Predicted LCMS Method"] = str(predicted_LCMS_method)
            print(f"LCMS prediction succesful...\n Predicted {predicted_LCMS_method} LCMS method for entry {i}")

            df = prediction_df.merge(descriptors_df, left_index=True, right_index=True)

        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

        return table

    

if __name__ == '__main__':
    
    app.run_server(debug=False)
