import streamlit as st
import pandas as pd
import numpy as np
from ttide import t_tide
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statistics import mode
import io
from statsmodels.tsa.arima.model import ARIMA
import tide_functions as tf
from streamlit_folium import st_folium
import folium


# folium map
m = folium.Map(location = [6.528261, 3.399947], zoom_start=16)
folium.Marker(
    [6.528261, 3.399947],
    popup='Bariga Jetty',
    tooltip='Bariga Jetty'
).add_to(m)

# Function to convert string to boolean
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Streamlit app title
st.title("Tide Analysis and Prediction Using Harmonic Analysis and ARIMA Model")

col_1, col_2 = st.columns(2)
with col_1:
    st.image('pngwing.com (93).png', width= 300,caption='Tides')

with col_2:
    st_folium(m,width=300,height=300)



# File upload
uploaded_file = st.file_uploader("Load your CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    raw_data = pd.read_csv(uploaded_file)

    # Display raw data
    st.write("Raw Data:")
    st.write(raw_data.head())

    # Select time and tidal value columns
    time_column = st.selectbox('Select Time Column', raw_data.columns)
    tidal_column = st.selectbox('Select Tidal Value Column', raw_data.columns)

    # Convert raw data to a dictionary
    raw_dict = raw_data.to_dict(orient='list')

    def inputDict1(time_col, tidal_col):
        '''
        Dictionary 1 containing pre-processed time and depth values and time interval between records.
        Processing initial input value from Main Widget
        '''
        data = pd.DataFrame(raw_dict).copy()

        # 'Date_Time' is the column you are converting to datetime
        time_column_name = time_col

        # Specify the datetime format based on your data
        datetime_format = "%m/%d/%Y %H:%M"  # Adjust this format according to your data

        # Convert the 'time' column to datetime with the specified format
        data[time_column_name] = pd.to_datetime(data[time_column_name], format=datetime_format)
        data.index = data[time_column_name]
        data = data.sort_index()

        time_array = data.index
        start_time = time_array[:-1]
        end_time = time_array[1:]
        time_diff_array = end_time - start_time
        time_diff = mode(time_diff_array)
        time_diff_float = time_diff.total_seconds() / 60  # Convert to minutes

        time_gap_div = np.where((time_diff_array > time_diff) & ((time_diff_array % time_diff) / time_diff == 0))
        time_gap_undiv = np.where((time_diff_array > time_diff) & ((time_diff_array % time_diff) / time_diff != 0))

        start_gap_div = start_time[time_gap_div] + time_diff
        end_gap_div = end_time[time_gap_div] - time_diff

        start_gap_undiv = start_time[time_gap_undiv] + time_diff
        end_gap_undiv = end_time[time_gap_undiv]

        data_dummy = []

        for i in range(len(start_gap_div)):
            if len(time_gap_div[0]) == 0:
                pass
            else:
                time_add = pd.date_range(start=start_gap_div[i], end=end_gap_div[i], freq=f'{time_diff.total_seconds()/60}T')
                nan_add = pd.DataFrame({time_column_name: time_add, tidal_col: pd.Series(np.nan, index=list(range(len(time_add))))})
                nan_add.index = nan_add[time_column_name]
                nan_add = nan_add.iloc[:, 1:]
                data_dummy.append(nan_add)

        for i in range(len(start_gap_undiv)):
            if len(time_gap_undiv[0]) == 0:
                pass
            else:
                time_add = pd.date_range(start=start_gap_undiv[i], end=end_gap_undiv[i], freq=f'{time_diff.total_seconds()/60}T')
                nan_add = pd.DataFrame({time_column_name: time_add, tidal_col: pd.Series(np.nan, index=list(range(len(time_add))))})
                nan_add.index = nan_add[time_column_name]
                nan_add = nan_add.iloc[:, 1:]
                data_dummy.append(nan_add)

        if len(data_dummy) > 0:
            data_add = pd.concat(data_dummy, sort=True)
            filled = pd.concat([data, data_add], sort=True)
        else:
            filled = data.copy()

        filled = filled.sort_index()
        time_array2 = filled.index
        depth_array2 = filled[tidal_col].values

        input_dict = {'depth': depth_array2, 'time': time_array2, 'interval': time_diff_float}
        return input_dict

    def ttideAnalyse(time_col, tidal_col, latitude):
        '''T Tide Analysis processing'''
        input_dict = inputDict1(time_col, tidal_col)
        ad = input_dict['depth']
        at = input_dict['time']
        time_diff = input_dict['interval'] / 60
        time_num = date2num(at.to_pydatetime())
        coef = t_tide(ad, dt=time_diff, stime=time_num[0], lat=latitude, synth=0)
        return coef

    def ttidePredict(start_date, end_date, interval, time_col, tidal_col, latitude):
        '''T Tide Prediction processing'''
        input_dict = inputDict1(time_col, tidal_col)
        coef = ttideAnalyse(time_col, tidal_col, latitude)
        time_num = date2num(input_dict['time'].to_pydatetime())
        msl = coef['z0']
        predic_time = pd.date_range(start=start_date, end=end_date, freq=interval)
        predic_time_num = date2num(predic_time.to_pydatetime())
        predic = coef(predic_time_num) + msl
        prediction_df = pd.DataFrame({'Time': predic_time, 'Predicted Tide': predic})
        st.write(f"Tide Predictions from {start_date} to {end_date}:")
        st.write(prediction_df)
        st.line_chart(prediction_df.set_index('Time')['Predicted Tide'])
        plt.figure(figsize=(12, 6))
        plt.plot(predic_time, predic, label='Predicted Tide')
        plt.xlabel('Time')
        plt.ylabel('Tide Level')
        plt.axhline(msl, color='r', label='MSL = ' + str(msl))
        plt.title(f'Tide Predictions from {start_date} to {end_date}')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        return prediction_df
    


    # Tab for user inputs
    tab1, tab2, tab3= st.tabs(["Tide Data", "Prediction Parameters", "ARIMA"])

    with tab1:
        st.write("Select Columns for Time and Tidal Data")

        if st.checkbox('Plot Observed Data'):
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(raw_data[time_column]), raw_data[tidal_column], label='Observed Tide')
            plt.xlabel('Time')
            plt.ylabel('Tide Level')
            plt.title('Observed Tide Data')
            plt.legend()
            plt.grid()
            st.pyplot(plt)

    with tab2:
        st.write("Input Prediction Parameters")
        latitude = st.number_input("Enter Latitude:", min_value=-90.0, max_value=90.0, value=6.514)
        start_date = st.date_input("Select Start Date:")
        end_date = st.date_input("Select End Date:")
        interval_options = {"1 Hour": 'H', "30 Minutes": '30T', "15 Minutes": '15T'}
        interval = st.selectbox("Select Prediction Interval", list(interval_options.keys()))

        if st.button("Predict Tides"):
            prediction_df = ttidePredict(start_date, end_date, interval_options[interval], time_column, tidal_column, latitude)

            # Save TTide report
            if st.button("Save TTide Report"):
                buffer = io.StringIO()
                prediction_df.to_csv(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="Download TTide Report",
                    data=buffer,
                    file_name="TTide_Report.csv",
                    mime="text/csv"
                )
    with tab3:
        st.write('Predict Tides using ARIMA Model')
        
        if st.button('Stationary Test'):
            st.write(tf.stationarity_test(raw_data))


        if st.button('Plot ACF and PACF'):
            plot_acf(raw_data.iloc[:,1], title='ACF Plot')
            plot_pacf(raw_data.iloc[:,1], title='PACF Plot')
            
            st.pyplot(plt)

        if st.button('Predict'):
            x_train, x_test = tf.train_test(raw_data)
            tf.model_training(x_train,x_test)

            st.pyplot(plt)

            
