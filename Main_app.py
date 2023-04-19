import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


def load_data():
    df = pd.read_csv('ma_lga_12345.csv')
    df['saledate'] = pd.to_datetime(df['saledate'], dayfirst=True)
    df1 = df.loc[(df['type'] == 'unit') & (df['bedrooms'] == 1)].copy()
    df1['Quarter'] = pd.PeriodIndex(df1['saledate'], freq='Q').strftime('%Y-Q%q')
    print(df1.head())  # Check if 'Quarter' column is added to df1
    df2 = df.loc[(df['type'] == 'house') & (df['bedrooms'] == 2)].copy()
    df2['Quarter'] = pd.PeriodIndex(df2['saledate'], freq='Q').strftime('%Y-Q%q')
    print(df2.head())  # Check if 'Quarter' column is added to df2
    df3 = df.loc[(df['type'] == 'house') & (df['bedrooms'] == 3)].copy()
    df3['Quarter'] = pd.PeriodIndex(df3['saledate'], freq='Q').strftime('%Y-Q%q')
    print(df3.head())
    df4 = df.loc[(df['type'] == 'house') & (df['bedrooms'] == 4)].copy()
    df4['Quarter'] = pd.PeriodIndex(df4['saledate'], freq='Q').strftime('%Y-Q%q')
    print(df4.head())
    df5 = df.loc[(df['type'] == 'house') & (df['bedrooms'] == 5)].copy()
    df5['Quarter'] = pd.PeriodIndex(df5['saledate'], freq='Q').strftime('%Y-Q%q')
    print(df5.head())
    ts1 = df1.set_index(['Quarter', 'bedrooms'])['MA'].to_frame()
    ts2 = df2.set_index(['Quarter', 'bedrooms'])['MA'].to_frame()
    ts3 = df3.set_index(['Quarter', 'bedrooms'])['MA'].to_frame()
    ts4 = df4.set_index(['Quarter', 'bedrooms'])['MA'].to_frame()
    ts5 = df5.set_index(['Quarter', 'bedrooms'])['MA'].to_frame()
    return ts1, ts2, ts3, ts4, ts5

ts1, ts2, ts3, ts4, ts5 = load_data()



######################### Define ARIMA functions
# fit ARIMA models for each time series
bestfit_log1 = ARIMA(ts1['MA'].apply(np.log), order=(0, 1, 3)).fit()
bestfit_log2 = ARIMA(ts2['MA'].apply(np.log), order=(0, 1, 0)).fit()
bestfit_log3 = ARIMA(ts3['MA'].apply(np.log), order=(4, 1, 0)).fit()
bestfit_log4 = ARIMA(ts4['MA'].apply(np.log), order=(2, 1, 0)).fit()
bestfit_log5 = ARIMA(ts5['MA'].apply(np.log), order=(1, 1, 1)).fit()

# extract fitted values
fitted_values1 = bestfit_log1.fittedvalues
fitted_values2 = bestfit_log2.fittedvalues
fitted_values3 = bestfit_log3.fittedvalues
fitted_values4 = bestfit_log4.fittedvalues
fitted_values5 = bestfit_log5.fittedvalues

# create new dataframes with fitted values
fitted_df1 = ts1.copy()
fitted_df1['fitted_values'] = np.exp(fitted_values1)
fitted_df2 = ts2.copy()
fitted_df2['fitted_values'] = np.exp(fitted_values2)
fitted_df3 = ts3.copy()
fitted_df3['fitted_values'] = np.exp(fitted_values3)
fitted_df4 = ts4.copy()
fitted_df4['fitted_values'] = np.exp(fitted_values4)
fitted_df5 = ts5.copy()
fitted_df5['fitted_values'] = np.exp(fitted_values5)
print(fitted_df1.head())

def plot_fitted_values(augmented_df, title, ylabel):
  fig, ax = plt.subplots(figsize=(10, 6))
  fig.autofmt_xdate()
  ax.plot(augmented_df.index.get_level_values(0), augmented_df['MA'], color='black', label='Data')
  ax.plot(augmented_df.index.get_level_values(0), augmented_df['fitted_values'], color='blue', label='Fitted')
  ax.set_xlabel('Quarter')
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.legend()
  plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right') # fixes the overlapping ticks issue
  plt.show()



plot_fitted_values(fitted_df1, "1BR Prices in Australia - Actual vs Predicted Values", "Price")
plot_fitted_values(fitted_df2, "2BR Prices in Australia - Actual vs Predicted Values", "Price")
plot_fitted_values(fitted_df3, "3BR Prices in Australia - Actual vs Predicted Values", "Price")
plot_fitted_values(fitted_df4, "4BR Prices in Australia - Actual vs Predicted Values", "Price")
plot_fitted_values(fitted_df5, "5BR Prices in Australia - Actual vs Predicted Values", "Price")


#################################################### main app.py
st.title('Welcome - Australian Housing Prices')
st.header('Actual vs Predicted Values by Number of Bedrooms')

st.set_option('deprecation.showPyplotGlobalUse', False)


menu = ['Home', 'Technical Background', 'About']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
  st.subheader('Welcome to the Australian Housing Prices')

  # Define a dropdown widget
  #type = st.selectbox('Type', ['unit', 'house'])
  #quarters = df['Quarter'].unique()
  #quarter = st.selectbox('Quarter', quarters)
  st.write('Please choose your desired number of bedroom(s):')
  bedroom_choice = st.selectbox('Number of Bedrooms', ['1', '2', '3', '4', '5'])
  st.write('Please choose when are you trying to sell/buy:')
  quarter = st.selectbox('Quarter', ['Any Quarter','Q1', 'Q2', 'Q3', 'Q4'])

  if bedroom_choice == '1':
    if quarter == 'Any Quarter':
      price = round(np.average(np.exp(fitted_values1)), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.subheader('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df1, "2BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 2 bedrooms:")
      st.write(fitted_df1)
    elif quarter == 'Q1':
      fitted_values1_Q1_all_years = round(np.exp(fitted_values1.loc[fitted_values1.index.get_level_values('Quarter').str.endswith('-Q1')]), 2)
      price = round(np.average(fitted_values1_Q1_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values1_Q1_all_years, "1BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 1 bedroom:")
      st.write(fitted_values1_Q1_all_years)
    elif quarter == 'Q2':
      fitted_values1_Q2_all_years = round(np.exp(fitted_values1.loc[fitted_values1.index.get_level_values('Quarter').str.endswith('-Q2')]), 2)
      price = round(np.average(fitted_values1_Q2_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See display and table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values1_Q2_all_years, "1BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 1 bedroom:")
      st.write(fitted_values1_Q2_all_years)
    elif quarter == 'Q3':
      fitted_values1_Q3_all_years = round(np.exp(fitted_values1.loc[fitted_values1.index.get_level_values('Quarter').str.endswith('-Q3')]), 2)
      price = round(np.average(fitted_values1_Q3_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See display and table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values1_Q3_all_years, "1BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 1 bedroom:")
      st.write(fitted_values1_Q3_all_years)
    elif quarter == 'Q4':
      fitted_values1_Q4_all_years = round(np.exp(fitted_values1.loc[fitted_values1.index.get_level_values('Quarter').str.endswith('-Q4')]), 2)
      price = round(np.average(fitted_values1_Q4_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See display and table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values1_Q4_all_years, "1BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 1 bedroom:")
      st.write(fitted_values1_Q4_all_years)

  elif bedroom_choice == '2':
    if quarter == 'Any Quarter':
      price = round(np.average(np.exp(fitted_values2)), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.subheader('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df2, "2BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 2 bedrooms:")
      st.write(fitted_df2)
    elif quarter == 'Q1':
      fitted_values2_Q1_all_years = round(np.exp(fitted_values2.loc[fitted_values2.index.get_level_values('Quarter').str.endswith('-Q1')]), 2)
      price = round(np.average(fitted_values2_Q1_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values2_Q1_all_years, "2BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 2 bedroom:")
      st.write(fitted_values2_Q1_all_years)
    elif quarter == 'Q2':
      fitted_values2_Q2_all_years = round(np.exp(fitted_values2.loc[fitted_values2.index.get_level_values('Quarter').str.endswith('-Q2')]), 2)
      price = round(np.average(fitted_values2_Q2_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See display and table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values2_Q2_all_years, "2BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 2 bedroom:")
      st.write(fitted_values2_Q2_all_years)
    elif quarter == 'Q3':
      fitted_values2_Q3_all_years = round(np.exp(fitted_values2.loc[fitted_values2.index.get_level_values('Quarter').str.endswith('-Q3')]), 2)
      price = round(np.average(fitted_values2_Q3_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See display and table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values2_Q3_all_years, "2BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 2 bedroom:")
      st.write(fitted_values2_Q3_all_years)
    elif quarter == 'Q4':
      fitted_values2_Q4_all_years = round(np.exp(fitted_values2.loc[fitted_values2.index.get_level_values('Quarter').str.endswith('-Q4')]), 2)
      price = round(np.average(fitted_values2_Q4_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See display and table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_values2_Q4_all_years, "1BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 2 bedroom:")
      st.write(fitted_values2_Q4_all_years)

  elif bedroom_choice == '3':
    if quarter == 'Any Quarter':
      price = round(np.average(np.exp(fitted_values3)), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df3, "3BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 3 bedrooms:")
      st.write(fitted_df2)
    if quarter == 'Q1':
      fitted_values3_Q1_all_years = round(np.exp(fitted_values3.loc[fitted_values3.index.get_level_values('Quarter').str.endswith('-Q1')]), 2)
      price = round(np.average(fitted_values3_Q1_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df3, "3BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 3 bedrooms:")
      st.write(fitted_values3_Q1_all_years)
    if quarter == 'Q2':
      fitted_values3_Q2_all_years = round(np.exp(fitted_values3.loc[fitted_values3.index.get_level_values('Quarter').str.endswith('-Q2')]), 2)
      price = round(np.average(fitted_values3_Q2_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df3, "3BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 3 bedrooms:")
      st.write(fitted_values3_Q2_all_years)
    if quarter == 'Q3':
      fitted_values3_Q3_all_years = round(np.exp(fitted_values3.loc[fitted_values3.index.get_level_values('Quarter').str.endswith('-Q3')]), 2)
      price = round(np.average(fitted_values3_Q3_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df3, "3BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 3 bedrooms:")
      st.write(fitted_values3_Q3_all_years)
    if quarter == 'Q4':
      fitted_values3_Q4_all_years = round(np.exp(fitted_values3.loc[fitted_values3.index.get_level_values('Quarter').str.endswith('-Q4')]), 2)
      price = round(np.average(fitted_values3_Q4_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df3, "3BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 3 bedrooms:")
      st.write(fitted_values3_Q4_all_years)

  elif bedroom_choice == '4':
    if quarter == 'Any Quarter':
      price = round(np.average(np.exp(fitted_values4)), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df4, "4BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 4 bedrooms:")
      st.write(fitted_df4)
    if quarter == 'Q1':
      fitted_values4_Q1_all_years = round(np.exp(fitted_values4.loc[fitted_values4.index.get_level_values('Quarter').str.endswith('-Q1')]), 2)
      price = round(np.average(fitted_values4_Q1_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df4, "4BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 4 bedrooms:")
      st.write(fitted_values4_Q1_all_years)
    if quarter == 'Q2':
      fitted_values4_Q2_all_years = round(np.exp(fitted_values4.loc[fitted_values4.index.get_level_values('Quarter').str.endswith('-Q2')]), 2)
      price = round(np.average(fitted_values4_Q2_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df4, "4BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 4 bedrooms:")
      st.write(fitted_values4_Q2_all_years)
    if quarter == 'Q3':
      fitted_values4_Q3_all_years = round(np.exp(fitted_values3.loc[fitted_values3.index.get_level_values('Quarter').str.endswith('-Q3')]), 2)
      price = round(np.average(fitted_values4_Q3_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df4, "4BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 4 bedrooms:")
      st.write(fitted_values4_Q3_all_years)
    if quarter == 'Q4':
      fitted_values4_Q4_all_years = round(np.exp(fitted_values4.loc[fitted_values4.index.get_level_values('Quarter').str.endswith('-Q4')]), 2)
      price = round(np.average(fitted_values4_Q4_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df4, "4BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 4 bedrooms:")
      st.write(fitted_values4_Q4_all_years)

  elif bedroom_choice == '5':
    if quarter == 'Any Quarter':
      price = round(np.average(np.exp(fitted_values5)), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df5, "5BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 5 bedrooms:")
      st.write(fitted_df5)
    if quarter == 'Q1':
      fitted_values5_Q1_all_years = round(np.exp(fitted_values5.loc[fitted_values5.index.get_level_values('Quarter').str.endswith('-Q1')]), 2)
      price = round(np.average(fitted_values5_Q1_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df5, "5BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 5 bedrooms:")
      st.write(fitted_values5_Q1_all_years)
    if quarter == 'Q2':
      fitted_values5_Q2_all_years = round(np.exp(fitted_values5.loc[fitted_values5.index.get_level_values('Quarter').str.endswith('-Q2')]), 2)
      price = round(np.average(fitted_values5_Q2_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df5, "5BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 5 bedrooms:")
      st.write(fitted_values5_Q2_all_years)
    if quarter == 'Q3':
      fitted_values5_Q3_all_years = round(np.exp(fitted_values5.loc[fitted_values5.index.get_level_values('Quarter').str.endswith('-Q3')]), 2)
      price = round(np.average(fitted_values5_Q3_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df5, "5BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 5 bedrooms:")
      st.write(fitted_values5_Q3_all_years)
    if quarter == 'Q4':
      fitted_values5_Q4_all_years = round(np.exp(fitted_values5.loc[fitted_values5.index.get_level_values('Quarter').str.endswith('-Q4')]), 2)
      price = round(np.average(fitted_values5_Q4_all_years), 2)
      formatted_price = '{:,.2f}'.format(price)
      st.write(f'<p style="font-size:24px;position:relative;display:inline-block;">Here is the predicted price of your chosen house: $ <b>{formatted_price}</b></p>\
      <div style="position:relative;width:100%;height:4px;background-color:#e6e6e6;overflow:hidden;">\
      <div style="position:absolute;width:0%;height:4px;background-color:#000080;animation: expand 3s forwards infinite;"></div>\
      </div>\
      <style>@keyframes expand {{to {{width:100%;}}}}</style>', unsafe_allow_html=True)
      st.write('\n')
      st.write('\n')
      st.write('NOTE: See table of the historical data')
      #st.pyplot(plot_fitted_values(fitted_df5, "5BR Prices in Australia - Actual vs Predicted Values", "Price"))
      #st.write("Actual Price (MA) and Predicted Price (fitted) for 5 bedrooms:")
      st.write(fitted_values5_Q4_all_years)
elif choice == 'Technical Background':\
  st.subheader('What You Need to Know How You Got This Price')
elif choice == 'About':\
  st.subheader('About Us')
st.write('1. Samaneh Torkzadeh')
st.write('Currently a grad student at IU Bloomington focusing on Data Science - Data Visualisation Domain. The Time Series class taught me what AI can do in many fields and how important it is to the modern world. This web app shows you a sneek peek of its powerful techniques regarding Time Series tasks.')
st.write('\n')
st.write('2. Emma Holler')
st.write('\n')
st.write('3. Mohit Mathrani')
