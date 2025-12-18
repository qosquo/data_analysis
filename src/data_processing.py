import os
import pandas as pd
import pytz
import kagglehub


def load_data():
    us_weather_events_path = kagglehub.dataset_download("sobhanmoosavi/us-weather-events")
    us_weather_df = pd.read_csv(
        os.path.abspath(
            os.path.join(
                os.getcwd(),
                us_weather_events_path,
                'WeatherEvents_Jan2016-Dec2022.csv'
            )
        )
    )

    flight_delays_path = kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")
    flight_delays = pd.read_csv(
        os.path.abspath(
            os.path.join(
                os.getcwd(),
                flight_delays_path,
                'flights_sample_3m.csv'
            )
        )
    )
    return us_weather_df, flight_delays


def process_weather_data(us_weather_df):
    # Convert StartTime(UTC) and EndTime(UTC) to datetime objects
    us_weather_df['StartTime(UTC)'] = pd.to_datetime(us_weather_df['StartTime(UTC)'])
    us_weather_df['EndTime(UTC)'] = pd.to_datetime(us_weather_df['EndTime(UTC)'])

    # 1. Create numerical mapping for Severity column
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Severe': 3, 'Extreme': 4}
    us_weather_df['Severity_Num'] = us_weather_df['Severity'].map(severity_mapping)

    # 2. Calculate duration of each weather event in hours
    us_weather_df['Duration_Hours'] = (
        (us_weather_df['EndTime(UTC)'] - us_weather_df['StartTime(UTC)']).dt.total_seconds() / 3600
    )

    # 3. Convert UTC time to local dates based on timezone
    def convert_utc_to_local(row):
        """Convert UTC time to local time using timezone information."""
        if pd.isna(row['StartTime(UTC)']):
            return pd.NaT
        try:
            utc_time = row['StartTime(UTC)'].tz_localize('UTC')
            local_tz = pytz.timezone(row['TimeZone'])
            return utc_time.tz_convert(local_tz)
        except Exception:
            return pd.NaT

    us_weather_df['Local_Start_Time'] = us_weather_df.apply(convert_utc_to_local, axis=1)
    us_weather_df['Local_Start_Time'] = pd.to_datetime(
        us_weather_df['Local_Start_Time'], errors='coerce'
    )
    us_weather_df['Local_Date'] = us_weather_df['Local_Start_Time'].dt.date

    # 4. Create boolean flags for specific weather types
    significant_weather_types = [
        'Snow', 'Fog', 'Heavy Rain', 'Thunderstorm', 'Rain',
        'Light Rain', 'Sleet', 'Freezing Rain', 'Wintry Mix'
    ]

    for weather_type in significant_weather_types:
        col_name = f'Is_{weather_type.replace(" ", "")}'
        us_weather_df[col_name] = us_weather_df['Type'].apply(
            lambda x: weather_type in x.split(', ') if pd.notna(x) else False
        )

    return us_weather_df


def create_icao_iata_mapping():
    icao_iata_data = {
        'icao_code': ['KABE', 'KPHX', 'KSEA', 'KJFK', 'KLGA', 'KORD', 'KATL', 'KLAX', 'KDEN', 'KDCA', 'KBOS'],
        'iata_code': ['ABE', 'PHX', 'SEA', 'JFK', 'LGA', 'ORD', 'ATL', 'LAX', 'DEN', 'DCA', 'BOS']
    }
    icao_iata_mapping = pd.DataFrame(icao_iata_data)
    icao_iata_mapping.loc[len(icao_iata_mapping)] = ['K04V', '04V']  # Hypothetical mapping

    return icao_iata_mapping


def convert_weather_codes_to_iata(aggregated_weather_df, icao_iata_mapping):
    # Merge with ICAO-IATA mapping
    aggregated_weather_df = pd.merge(
        aggregated_weather_df,
        icao_iata_mapping,
        left_on='AirportCode',
        right_on='icao_code',
        how='left'
    )

    # Create temporary column for IATA codes, falling back to original ICAO if no mapping
    aggregated_weather_df['AirportCode_IATA_Temp'] = aggregated_weather_df['iata_code'].fillna(
        aggregated_weather_df['AirportCode']
    )

    # Drop intermediate columns
    aggregated_weather_df.drop(
        columns=['AirportCode', 'icao_code', 'iata_code'],
        inplace=True
    )
    aggregated_weather_df.rename(columns={'AirportCode_IATA_Temp': 'AirportCode'}, inplace=True)

    return aggregated_weather_df


def merge_flight_and_weather(aggregated_flight_df, aggregated_weather_df):
    # Convert date columns to date type for proper merging
    aggregated_flight_df['FL_DATE'] = aggregated_flight_df['FL_DATE'].dt.date
    aggregated_flight_df['ORIGIN'] = aggregated_flight_df['ORIGIN'].astype(str)
    aggregated_weather_df['AirportCode'] = aggregated_weather_df['AirportCode'].astype(str)

    # Merge dataframes
    merged_df = pd.merge(
        aggregated_flight_df,
        aggregated_weather_df,
        left_on=['ORIGIN', 'FL_DATE'],
        right_on=['AirportCode', 'Local_Date'],
        how='left'
    )

    # Drop redundant columns
    merged_df.drop(columns=['AirportCode', 'Local_Date'], inplace=True)

    # Define lists of weather and flight features
    weather_features = [
        'Max_Severity',
        'Total_Duration_Hours',
        'Num_Weather_Events',
        'Is_Snow',
        'Is_Fog',
        'Is_HeavyRain',
        'Is_Thunderstorm',
        'Is_Rain',
        'Is_LightRain',
        'Is_Sleet',
        'Is_FreezingRain',
        'Is_WintryMix'
    ]
    flight_metrics = [
        'Total_Cancelled_Flights',
        'Average_Dep_Delay',
        'Total_Weather_Delays'
    ]

    # Identify boolean columns for filling NaNs with False
    boolean_weather_cols = [col for col in weather_features if 'Is_' in col]
    # Identify numerical columns for filling NaNs with 0
    numerical_weather_cols = [col for col in weather_features if col not in boolean_weather_cols]

    # Create a copy to avoid SettingWithCopyWarning
    correlation_prep_df = merged_df.copy()

    # Fill NaN values in boolean weather columns with False
    for col in boolean_weather_cols:
        if col in correlation_prep_df.columns:
            correlation_prep_df[col] = correlation_prep_df[col].fillna(False).astype(bool)

    # Fill NaN values in numerical weather columns with 0
    for col in numerical_weather_cols:
        if col in correlation_prep_df.columns:
            correlation_prep_df[col] = correlation_prep_df[col].fillna(0)

    return correlation_prep_df


if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    # Load raw data
    print("Loading data...")
    us_weather_df, flight_delays = load_data()

    # Process data
    print("Processing weather data...")
    us_weather_df = process_weather_data(us_weather_df)

    # Convert 'FL_DATE' to datetime objects
    print("Processing flight data...")
    flight_delays['FL_DATE'] = pd.to_datetime(flight_delays['FL_DATE'])

    # Aggregate data
    print("Aggregating weather data...")
    significant_weather_types = [
        'Snow', 'Fog', 'Heavy Rain', 'Thunderstorm', 'Rain',
        'Light Rain', 'Sleet', 'Freezing Rain', 'Wintry Mix'
    ]

    agg_dict = {
        'Max_Severity': ('Severity_Num', 'max'),
        'Total_Duration_Hours': ('Duration_Hours', 'sum'),
        'Num_Weather_Events': ('EventId', 'count'),
    }

    # Add weather type flags to aggregation dictionary
    for weather_type in significant_weather_types:
        col_name = f'Is_{weather_type.replace(" ", "")}'
        agg_dict[col_name] = (col_name, 'any')

    aggregated_weather_df = us_weather_df.groupby(['AirportCode', 'Local_Date']).agg(
        **{key: value for key, value in agg_dict.items()}
    ).reset_index()

    print("Aggregating flight data...")
    aggregated_flight_df = flight_delays.groupby(['ORIGIN', 'FL_DATE']).agg(
        Total_Flights=('FL_NUMBER', 'count'),
        Total_Cancelled_Flights=('CANCELLED', lambda x: (x == 1).sum()),
        Average_Dep_Delay=('DEP_DELAY', 'mean'),
        Total_Weather_Delays=('DELAY_DUE_WEATHER', lambda x: (x == 1).sum())
    ).reset_index()

    # Convert airport codes and merge
    print("Converting ICAO codes to IATA codes...")
    icao_iata_mapping = create_icao_iata_mapping()
    aggregated_weather_df = convert_weather_codes_to_iata(aggregated_weather_df, icao_iata_mapping)

    print("Merging flight and weather data...")
    merged_df = merge_flight_and_weather(aggregated_flight_df, aggregated_weather_df)
    # Save merged_df to csv
    print("Saving merged data to CSV...")
    path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'merged_flight_weather_data.csv'))
    merged_df.to_csv(path, index=False)

    print("Data processing complete!")