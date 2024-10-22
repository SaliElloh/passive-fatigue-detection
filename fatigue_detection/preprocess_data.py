import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv

RandomForestModel_dir = 'C:/Users/sally/Documents/GitHub/new_transfer_learning/fatigue_detection/random_forest_model.pkl'

scaler = StandardScaler()

labels = ['inattentive', 'imp_attentive', 'unimp_attentive']

model = joblib.load(RandomForestModel_dir)

WEATHER_CLASSES =  ['Evening' ,'Morning', 'Night',
                    'Cloudy', 'Rainy','Sunny', 
                    'Countryside', 'Downtown', 'Highway']

def event_counter(df):
    """
    Analyzes consecutive occurrences of events within a DataFrame.

    This function calculates two things for each event type in the DataFrame:
    1. The length of consecutive sequences of each event type.
    2. The number of times each event type appears consecutively in these sequences.

    Args:
        df (pd.DataFrame): DataFrame containing at least a column named 'event_type'.

    Returns:
        pd.Series: A Series with the count of consecutive occurrences of each event type.
    """
    result = df.groupby(df['event_type'].ne(df['event_type'].shift()).cumsum())['event_type'].value_counts() 
    result = pd.DataFrame(result)
    # This line had an indentation error

    return result

def calculate_saccade_metrics(df):
    """
    Calculates the distance and velocity for saccades in a DataFrame and adds the metrics to the original DataFrame.

    Args:
      df: A pandas DataFrame containing saccade data.

    Returns:
      The original DataFrame with distance and velocity columns added for saccades.
    """
    
    # Make a copy of the DataFrame to avoid modifying the original one
    df_new = df.copy()
    
    # Perform the calculations on the filtered saccade rows
    df_new["time_dif"] = df_new['timeinsec'].diff()
    df_new['x'] = pd.to_numeric(df_new['x'], errors='coerce')
    df_new['y'] = pd.to_numeric(df_new['y'], errors='coerce')
    df_new['distance'] = np.sqrt((df_new['x'].diff().fillna(0) ** 2) + (df_new['y'].diff().fillna(0) ** 2))

    # Replace zero or NaN time differences with 1/30
    df_new['time_dif'] = df_new['time_dif'].replace({0: 1/30, np.nan: 1/30})
    
    # Calculate velocity
    df_new['velocity'] =df_new['distance'] /df_new['time_dif']

    # Update the original DataFrame with the calculated metrics
    df_new.update(df_new[['distance', 'velocity']])
    
    # Return the updated DataFrame with new metrics added
    return df_new


def preprocess_weather_label(arr):
    return [1 if weather_class in arr else 0 for weather_class in WEATHER_CLASSES]

            
def predict_label(five_min_data):
    # Convert the input to a DataFrame with the specified columns
    df = pd.DataFrame(five_min_data, columns=['x',
                                               'y', 
                                               'timeinsec',
                                                 'event_type', 
                                                 'time_Evening' ,
                                                 'time_Morning' ,
                                                 'time_Night',
                                                 'weather_Cloudy',
                                                 'weather_Rainy',
                                                   'weather_Sunny', 
                                                   'location_Countryside', 
                                                   'location_Downtown', 
                                                   'location_Highway'])

    # # Encode the 'event_type' column into dummy variables
    df_encoded = pd.get_dummies(df, columns=['event_type'], dtype=int)
    df_encoded.index = range(0, len(df_encoded))
    print(df_encoded.columns)

    # Scale the data using the scaler (assuming scaler is pre-defined)
    scaled_data = scaler.fit_transform(df_encoded)

    # Make predictions using the pre-trained model (assuming the model is pre-defined)
    predictions = model.predict(scaled_data)

    # Get the predicted labels from the model's output
    predictions_arr = np.array(predictions)
    predicted_labels = [labels[np.argmax(pred)] for pred in predictions_arr]

    predicted_labels_df = pd.DataFrame({'timeinsec': df['timeinsec'],
                              'predictions': predicted_labels                  
                              }
                             )

    return predicted_labels_df
