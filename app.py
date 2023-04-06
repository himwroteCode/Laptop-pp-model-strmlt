import streamlit as st
import os
import time
import pickle
import pandas as pd
#import streamlit_option_menu
#from streamlit_option_menu import option_menu

st.set_page_config(page_title="AboutThisApp",
                   layout="wide" 
                   )

# Get the absolute path of the files
org_df = os.path.abspath(os.path.join(os.path.dirname(__file__), "resources", "data", "laptop_details.csv"))
cleaned_df = os.path.abspath(os.path.join(os.path.dirname(__file__), "resources", "data", "df2.csv"))

# Load the laptop details data from the csv file
laptop_details = pd.read_csv(org_df)
df2 = pd.read_csv(cleaned_df)



with st.sidebar:
    selected=option_menu(
         menu_title="App content Guide",
         options=['About this app', 'Laptop Market Analysis', 'Laptop Value', 'About Author'])


if selected== "About this app":
    st.header('App Manual')

    st.balloons()
    st.subheader(" This app will let you know the :orange[market value] of a laptop in :blue[India] ")
    st.write("In the `Laptop Market Analysis` tab, you can find information about market laptop in India ")
    st.write("In the `Laptop Value` tab, you can go and check for the price  of a laptop of your desired specification ")
    st.write("In the `About Author` tab, you can find information about who developed this app ")
    st.subheader("Here you can find the original data that was scrapped from web")

    st.subheader('Original Data')
    st.write(laptop_details)

    st.subheader("This, the below :red[data] is extracted from the :blue[original] to so that predictions could be made")
    st.subheader("Cleaned Data")
    st.write(df2)
    st.write("This above data is  used to  predict the `Laptop Value` of your given specification")

if selected=='Laptop Market Analysis':
    st.snow()
    st.header("Laptop market analysis in India")
    st.write("""   
    
    :blue[Introduction:]

The laptop market has grown exponentially in recent years, thanks to the increasing demand for portable computing devices.
The market is highly competitive, with several manufacturers vying for a larger market share. In this report,
we will analyze the laptop market and discuss the factors that affect the price of laptops, including RAM, OS, processor, brand, and more.

:red[RAM:]

Random Access Memory (RAM) is an essential component of any laptop as it affects its performance significantly. A laptop with higher
RAM can handle multiple applications and processes simultaneously, making it more efficient. However, higher RAM also increases the cost
of the laptop. In general, laptops with 8GB to 16GB RAM are most popular and cost-effective for regular users, while laptops with 32GB
or more RAM are usually more expensive and targeted towards power users or gamers.

:blue[OS:]

The Operating System (OS) installed on a laptop can affect its price as well. Laptops with pre-installed Windows OS usually cost more
than those with Linux or Chrome OS. This is because Windows OS is more widely used and provides more features than other operating systems.
However, the price difference may not be significant, and many users opt to install their preferred OS, making this factor less important.

:red[Processor:]

The processor is the heart of a laptop and is a crucial component in determining its price. Intel and AMD are the most popular processor
manufacturers, and their processors come in different models and generations. Higher-end processors such as Intel Core i9 or AMD Ryzen 9
cost more than lower-end processors like Intel Core i3 or AMD Ryzen 3. The processor's generation also affects its price, with newer
generations costing more than older ones.

:green[Brand:]

The brand of a laptop also affects its price. Popular brands like Apple, Dell, HP, and Lenovo usually charge a premium for their laptops
compared to lesser-known brands. This is because popular brands have established their reputation for quality and reliability, and their
laptops come with additional features and services such as customer support, warranty, and after-sales services.

:blue[Storage:]

Storage capacity is another crucial factor that affects the laptop's price. Laptops with larger storage capacity such as 1TB or more are
usually more expensive than those with 256GB or 512GB storage. The type of storage also affects the price, with Solid State Drives (SSD)
being more expensive than Hard Disk Drives (HDD). However, SSDs are faster and more durable than HDDs, making them the preferred storage
option for most users.

:red[Conclusion:]

In conclusion, several factors affect the price of laptops, including RAM, OS, processor, brand, and storage. Users need to balance their
needs and budget while choosing a laptop that suits them. In general, laptops with 8GB to 16GB RAM, 256GB to 512GB SSD, Intel Core i5 or
AMD Ryzen 5 processors, and a popular brand like Dell or HP are the most cost-effective and suitable for regular users. However, power
users or gamers may require higher-end laptops with 32GB or more RAM, 1TB or more SSD, Intel Core i9 or AMD Ryzen 9 processors, and a
dedicated graphics card.
    
    """)
#############################################################################################
import sklearn
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer as ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

import os
import pickle

# set the file paths for the pickled objects
data_path = os.path.join("D:", "innomatics_ds_internship", "Laptop Price Prediction streamlit webapp", "resources", "data", "df22.pkl")
model_path = os.path.join("D:", "innomatics_ds_internship", "Laptop Price Prediction streamlit webapp", "resources", "data", "model2.pkl")

 # check if the files exist
if os.path.isfile(data_path) and os.path.isfile(model_path):
     #load the pickled objects using the pickle module
    with open(data_path, 'rb') as f:
        lap = pickle.load(f)

    with open(model_path, 'rb') as f:
        rf = pickle.load(f)
    df = pd.DataFrame(lap) 
    # do further processing with the loaded objects
    # ...
#else:
 #   print("File not found")




# st.dataframe(df)
# ----------------------------------------ML section------------------------------------------
features = ["brand","processor_type", "ram","storage","os"]
f = df2[[ "brand","processor_type", "ram","storage","os"]]
y = np.log(df2['MRP'])
X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=47)
step1 = ct(transformers=[
    ('encoder',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,4])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

if selected=='Laptop Value':
    st.snow()
    st.header("Select specification of laptop that you want to know price of ")
# -----------------------------------Input Section---------------------------------------------
    brand = st.selectbox("Select Brand:- ", df2["brand"].unique())
    processor_type = st.selectbox("Select Processor:- ", df2["processor_type"].unique())
    ram = st.selectbox("Select the RAM:- ", df2["ram"].unique())
    storage = st.selectbox("Select the Storage(ssd- 128, 256, 512 and Hdd- 1Tb, 2Tb):- ", df2["storage"].unique())
    os = st.selectbox("Select the Operating Syatem:- ", df2["os"].unique())
#display_size = st.selectbox("Select the display size of System:- ", df2["display_size"].unique())
    st.write("Do You wanna Predict the Price of the Laptop ❓")
    butt = st.button("Predict ❗")
    if butt:
        st.snow()
        query = np.array([brand, processor_type, ram, storage, os])
        query = query.reshape(1, -1)
        p = pipe.predict(query)[0]
        result = np.exp(p)
        st.subheader("Your Predicted Price is: ")
        st.subheader(":red[₹{}]".format(result.round(2)))

        st.subheader("Check for different specification")

# -------------------------------------The Image Generation Section--------------------------

if selected=='About Author':
    st.header("About myself")
    st.write("""I am :blue[Himanshu Vaish], I have given a task to make a laptop price prediction model based upon given data.
    In this project I have used ColumnTransformer from sklearn library and encoded the input value at runtime. This project is predicting price 
    on comparitively less features and  number of row was also not really much  as you can see in `About page`, so acuracy might be different and
    it might  also happen that some basic things may also not work.
    `If you are reading this, that means I have not fixed the issue but I am working hard to fix this and I'll do it soon`.
    If you have any  question/suggestion/query then please connect me anywhere in the links given below.
    :blue[Thank you]
    
    """)
    st.write(":red[My GitHub profile is] [here](https://github.com/himwroteCode).")
    st.write(":blue[My Linkedin profile is] [here](https://www.linkedin.com/in/vaishhimanshu/).")

st.sidebar.empty()
