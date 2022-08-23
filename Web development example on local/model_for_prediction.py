# import all of the modules used
import pickle
import pandas as pd
import os

model_save_path = 'XGB_final_model.pkl'
# open pickled model
with open(model_save_path, 'rb') as file:
    unpickled_model = pickle.load(file)


def upload_data(filename):
    """
    Function is used calculate the response variable from the raw data of
    the EDSA Individual | Electricity Shortfall Challenge

    input data: Raw EDSA Individual | Electricity Shortfall Challenge CSV Data provided by user
    output: time and response variable calculated from input data as a csv file.
    :return: name of the output csv file: final_submission_result.csv
    """

    # Read csv file provided
    df_file = pd.read_csv(filename)
    # Convert data provided to a dataframe
    df_original = pd.DataFrame(df_file)
    # Create copy of original dataframe created
    df = df_original.copy()

    # Update missing data
    df['Valencia_pressure'] = df['Valencia_pressure'].fillna(1018.0)
    # Add new features
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    df['week'] = df['time'].dt.week
    # Convert all objects to a numeric value
    df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.extract('(\d+)')
    df['Seville_pressure'] = df['Seville_pressure'].str.extract('(\d+)')
    df['Valencia_wind_deg'] = pd.to_numeric(df['Valencia_wind_deg'])
    df['Seville_pressure'] = pd.to_numeric(df['Seville_pressure'])
    # Remove all columns not needed for model prediction
    df = df.drop(['Unnamed: 0', 'time'], axis=1)
    # Ensure the order of columns in DataFrame matches the data used to train the model
    column_order = ['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
                    'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
                    'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
                    'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
                    'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
                    'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
                    'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
                    'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
                    'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
                    'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure',
                    'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
                    'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
                    'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
                    'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
                    'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min',
                    'year', 'month', 'day', 'hour', 'weekday', 'week']

    df = df[column_order]

    # import the final prediction model saved with pickle
    model_save_path = 'XGB_final_model.pkl'
    # open pickled model
    with open(model_save_path, 'rb') as file:
        unpickled_model = pickle.load(file)

    # Use model Predict the response variable of the cleaned unseen data
    # Create new dataframe for the results with load_shortfall_3h as response column header
    results = pd.DataFrame(unpickled_model.predict(df), columns=['load_shortfall_3h'])
    # Create dataframe of time column orignally dropped from database
    df_original = df_original[['time']]
    # Join the response variables and time dataframes
    submission = df_original.join(results)
    # Create file path and saved name for final submission document
    results_save_loc = f"{os.path.join('file_results', 'final_submission_result')}.csv"
    # Save the file to the file_results folder
    submission.to_csv(results_save_loc, index=False)
    # Return the final submission csv name
    return 'final_submission_result.csv'
