# streamlit-demo
Material for a course for Utvecklarakademin.

## 1) Set up environment
Install required libraries from requirements.txt in virtualenv or conda environment (virtual environments for Python - you generally dont want to install libraries in your global Python version).

## 2) Prepare dataset
Download the dataset from kaggle: https://www.kaggle.com/datasets/knightbearr/sales-product-data?select=Sales_September_2019.csv
Put it in the streamlit_demo/data folder. (Note that streamlit_demo is a subfolder of streamlit-folder, according to python conventional naming.)

## 3) Run the streamlit app to start the dashboard
Make sure you are in the root folder of the project (streamlit-demo) and run the following command in the terminal:
```bash
streamlit run streamlit_demo/app.py
```
The dashboards should start in your web browser. You can now use the slicers!
