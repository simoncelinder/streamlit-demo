# streamlit-demo
Material for a course for Utvecklarakademin.

## 1) Set up environment
Install required libraries from requirements.txt in virtualenv or conda environment (virtual environments for Python - you generally dont want to install libraries in your global Python version).
If you do it from the terminal, make sure your environment is active in that terminal window, and run
```pip install -r requirements.txt``` (make sure to be in the same folder as requirements.txt).

## 2) Prepare dataset
Download the dataset from open link in Google Drive: https://drive.google.com/file/d/1JyKz5waLTxIuov50W4TgHRwuP0vnE_kn/view?usp=share_link
Put it in the streamlit-demo/streamlit_demo/data folder.

(The data originally comes from kaggle, but everyone might not have Kaggle account): https://www.kaggle.com/datasets/knightbearr/sales-product-data?select=Sales_September_2019.csv)


## 3) Run the streamlit app to start the dashboard
Make sure you are in the root folder of the project (streamlit-demo) and run the following command in the terminal:
```bash
streamlit run streamlit_demo/app.py
```
The dashboards should start in your web browser. You can now use the slicers!
