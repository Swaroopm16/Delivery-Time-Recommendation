import numpy as np
import pandas as pd



def data_cleaning(data: pd.DataFrame):
    # 1. Strip hidden trailing/leading spaces from ALL text entries first
    string_cols = data.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        data[col] = data[col].str.strip()
        
    # 2. Globally replace the text "NaN" string with real python nulls
    data = data.replace("NaN", np.nan)
    
    # 3. Safely grab indices for outliers now that strings are handled
    minors_data = data.loc[data['age'].astype(float) < 18]
    minor_index = minors_data.index.tolist()
    
    six_star_data = data.loc[data['ratings'].astype(float) == 6.0]
    six_star_index = six_star_data.index.tolist()
    
    # 4. Run the main processing pipeline
    return (
        data
        .drop(columns="id", errors="ignore") # errors="ignore" prevents crashes if dropped already
        .drop(index=minor_index, errors="ignore")
        .drop(index=six_star_index, errors="ignore")
        .assign(
            # city column out of rider id
            city_name = lambda x: x['rider_id'].str.split("RES").str.get(0),
            # convert age and ratings explicitly to float
            age = lambda x: x['age'].astype(float),
            ratings = lambda x: x['ratings'].astype(float),
            # absolute values for location based columns
            restaurant_latitude = lambda x: x['restaurant_latitude'].astype(float).abs(),
            restaurant_longitude = lambda x: x['restaurant_longitude'].astype(float).abs(),
            delivery_latitude = lambda x: x['delivery_latitude'].astype(float).abs(),
            delivery_longitude = lambda x: x['delivery_longitude'].astype(float).abs(),
            # order date to datetime and feature extractio
 
            # time taken to pick order
            # hour in which order was placed
            # time of the day when order was placed
            # categorical columns
            weather = lambda x: (
                x['weather']
                .str.replace("conditions ", "")
                .str.lower()
                .replace("nan", np.nan)
            ),
            traffic = lambda x: x["traffic"].str.rstrip().str.lower(),
            type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),
            type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
            festival = lambda x: x['festival'].str.rstrip().str.lower(),
            city_type = lambda x: x['city_type'].str.rstrip().str.lower(),
            # multiple deliveries column
            multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),
            # target column modifications
            time_taken = lambda x: (
                x['time_taken']
                .astype(int)

            )
        )
    )
# %%
def clean_lat_long(data: pd.DataFrame, threshold=1):
    location_columns = [
        'restaurant_latitude',
        'restaurant_longitude',
        'delivery_latitude',
        'delivery_longitude'
    ]
    
    return (
        data
        .assign(**{
            col: (
                np.where(data[col] < threshold, np.nan, data[col].values)
            )
            for col in location_columns
        })
    )

# %%
# extract day, day name, month and year
def extract_datetime_features(ser):
    date_col = pd.to_datetime(ser, dayfirst=True)
    
    return (
        pd.DataFrame({
            "day": date_col.dt.day,
            "month": date_col.dt.month,
            "year": date_col.dt.year,
            "day_of_week": date_col.dt.day_name(),
            "is_weekend": date_col.dt.day_name().isin(["Saturday", "Sunday"]).astype(int)
        })
    )

# %%
def time_of_day(ser):
    return (
        pd.cut(
            ser, 
            bins=[0, 6, 12, 17, 20, 24], 
            right=True,
            labels=["after_midnight", "morning", "afternoon", "evening", "night"]
        )
    )

# %%
def calculate_haversine_distance(df):
    location_columns = [
        'restaurant_latitude',
        'restaurant_longitude',
        'delivery_latitude',
        'delivery_longitude'
    ]
    
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c
    
    return (
        df.assign(
            distance = distance
        )
    )
# %%
def calculate_haversine_distance(df):
    location_columns = [
        'restaurant_latitude',
        'restaurant_longitude',
        'delivery_latitude',
        'delivery_longitude'
    ]
    
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c
    
    return (
        df.assign(
            distance = distance
        )
    )

# %%
def create_distance_type(data: pd.DataFrame):
    return (
        data
        .assign(
            distance_type = pd.cut(
                data["distance"],
                bins=[0, 5, 10, 15, 25],
                right=False,
                labels=["short", "medium", "long", "very_long"]
            )
        )
    )

# %%
def perform_data_cleaning(data: pd.DataFrame, saved_data_path="swiggy_cleaned.csv"):
    cleaned_data = (
        data
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
    )
    
    # save the data
    cleaned_data.to_csv(saved_data_path, index=False)
    return cleaned_data