import streamlit as st
import pandas as pd
from PIL import Image
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import subprocess
import os
import base64
import pickle

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

def model_build_fn():
    model = Sequential()
    model.add(Dense(218, input_dim=218, kernel_initializer='normal', activation='relu'))
    model.add(Dense(650, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
  # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    history=model.fit(x_train, y_train, epochs=100, batch_size=150, verbose=1, validation_split=0.2)
    predictions = model.predict(x_val)  
    return model

# Model building
def build_model(input_data):
    # Reads in saved regression model
    model2 = KerasRegressor(build_fn=model_build_fn, epochs=10, batch_size=10, verbose=1)
    model2.model = load_model('saved_model.h5')
    # Apply model to make predictions
    prediction = model2.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction + 1.5, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Logo image
image = Image.open('logo-removebg-preview.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""
## Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.

**About**
- Computational drug discovery tool using QSAR modelling with Convolution Neural Networking.
---
""")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')
