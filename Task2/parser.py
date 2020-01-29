import pandas as pd


"""

Choose the k rows from the pandas dataframe that have 
the biggest values in column_name column.

k - number of top rows
frame - pandas dataframe to work with
column_name - column on which to sort values on
columns_to_keep - which columns from the original dataframe to return after sorting;
                if None - the original dataframe columns (k amount, sorted) will be returned
draft_city - optional, which column to use to fill the None values in the 'city' column
            (indexed 0 in columns_to_keep), if no columns_to_keep provided, then the value is ignored

"""

def choose_k_biggest(k, frame, column_name, columns_to_keep = None, draft_city = None):

    assert isinstance(frame, pd.DataFrame)
    frame_sorted = frame[frame[column_name].apply(lambda x: x.isnumeric())]
    frame_sorted = frame_sorted.astype({column_name: 'int'})
    frame_sorted = frame_sorted.sort_values(by = column_name, ascending=False)
    frame_sorted = frame_sorted.head(k)

    if columns_to_keep is not None:
        assert isinstance(columns_to_keep, list)
        assert set(columns_to_keep).issubset(frame_sorted.columns)
        if draft_city is not None:
            frame_sorted[columns_to_keep[0]].fillna(frame_sorted[draft_city] ,inplace = True)
        frame_sorted = frame_sorted[columns_to_keep]

    return frame_sorted


"""

Read and sort csv file
path - path to csv source
column_name - column on which to sort values on
columns_to_keep - which columns from the original dataframe to return after sorting;
                if None - the original dataframe columns (k amount, sorted) will be returned
draft_city - optional, which column to use to fill the None values in the 'city' column
            (indexed 0 in columns_to_keep), if no columns_to_keep provided, then the value is ignored
k - number of top rows
"""
def process_csv(path, column_name, columns_to_keep = None, draft_city = None,k = 30):

    df = pd.read_csv(path)

    df_sorted = choose_k_biggest(k, df, column_name, columns_to_keep, draft_city)


    return df_sorted


"""

Extract the coordinates and city names from pandas dataframe to list

frame - pandas df to work with
coords_names - columns with the coordinates, should be in the format (x_index, y_index) to compute euclidean distance,
                (lon_index, lat_index) to compute distance in lat/lon coordinates (lon_index is the first, 
                as this format is geopy lib specific)

"""
def convert_format(frame, coords_names, city_names):
    names = list(frame[city_names])
    frame['coords'] = list(zip(frame[coords_names[0]], frame[coords_names[1]]))
    coords = list(frame['coords'])

    return names, coords

