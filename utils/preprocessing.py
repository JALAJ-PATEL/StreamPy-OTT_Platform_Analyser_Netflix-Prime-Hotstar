import pandas as pd
import numpy as np
import re

def clean_duration(df):
    """
    Clean and extract numeric values from duration column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a duration column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added duration_value and duration_unit columns
    """
    df_copy = df.copy()
    
    # Check if duration column exists and is not empty
    if 'duration' not in df_copy.columns:
        # Create empty duration columns
        df_copy['duration'] = ''
        df_copy['duration_value'] = float('nan')
        df_copy['duration_unit'] = ''
        return df_copy
    
    # Handle missing values
    df_copy['duration'] = df_copy['duration'].fillna('')
    
    # Check if duration is a DataFrame (which would cause the error)
    if isinstance(df_copy['duration'], pd.DataFrame):
        # If it's a DataFrame, convert to Series first
        df_copy['duration'] = df_copy['duration'].iloc[:, 0]
    
    # Only process rows with non-empty duration strings
    mask = df_copy['duration'].astype(str).str.len() > 0
    if mask.any():
        duration_series = df_copy.loc[mask, 'duration']
        
        # Extract values and units
        df_copy.loc[mask, 'duration_value'] = duration_series.str.extract(r'(\d+)').astype(float)
        df_copy.loc[mask, 'duration_unit'] = duration_series.str.extract(r'(\D+)').str.strip()
    else:
        # No valid durations, create empty columns
        df_copy['duration_value'] = float('nan')
        df_copy['duration_unit'] = ''
    
    return df_copy

def standardize_ratings(df):
    """
    Standardize ratings across different platforms
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a rating column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with standardized ratings
    """
    df_copy = df.copy()
    
    # Check if rating column exists
    if 'rating' not in df_copy.columns:
        # Create rating columns
        df_copy['rating'] = ''
        df_copy['standardized_rating'] = ''
        return df_copy
    
    # Handle missing ratings
    df_copy['rating'] = df_copy['rating'].fillna('').astype(str)
    
    # Define rating mappings
    rating_mappings = {
        # Various versions of "Not Rated"
        'NR': 'Not Rated',
        'NOT RATED': 'Not Rated',
        'UR': 'Not Rated',
        'UNRATED': 'Not Rated',
        
        # General audience
        'G': 'G',
        'TV-G': 'G',
        'ALL': 'G',
        'ALL_AGES': 'G',
        
        # Parental guidance
        'PG': 'PG',
        'TV-PG': 'PG',
        
        # Teen and up
        'PG-13': 'PG-13',
        'TV-14': 'PG-13',
        '13+': 'PG-13',
        
        # Mature
        'R': 'R',
        'TV-MA': 'R',
        'MATURE': 'R',
        '16+': 'R',
        '17+': 'R',
        '18+': 'R',
        'NC-17': 'R'
    }
    
    # Apply mapping - handle empty ratings
    mask = df_copy['rating'].str.len() > 0
    if mask.any():
        df_copy.loc[mask, 'standardized_rating'] = df_copy.loc[mask, 'rating'].map(
            lambda x: rating_mappings.get(str(x).upper(), x)
        )
    else:
        df_copy['standardized_rating'] = ''
    
    return df_copy

def clean_text_columns(df):
    """
    Clean text columns (description, title, etc.)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with text columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cleaned text columns
    """
    df_copy = df.copy()
    
    # Text columns to clean
    text_columns = ['title', 'description', 'director', 'cast', 'country', 'listed_in']
    
    for col in text_columns:
        # If column exists, clean it
        if col in df_copy.columns:
            # Handle missing or non-string values
            df_copy[col] = df_copy[col].fillna('').astype(str)
            
            # Only process non-empty strings
            mask = df_copy[col].str.len() > 0
            if mask.any():
                # Remove leading/trailing whitespace
                df_copy.loc[mask, col] = df_copy.loc[mask, col].str.strip()
                
                # Replace multiple spaces with single space
                df_copy.loc[mask, col] = df_copy.loc[mask, col].str.replace(r'\s+', ' ', regex=True)
        else:
            # Create the column if it doesn't exist
            df_copy[col] = ''
    
    return df_copy

def extract_release_date_info(df):
    """
    Extract year and month information from date columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing date columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional date-related columns
    """
    df_copy = df.copy()
    
    # Process date_added column
    if 'date_added' in df_copy.columns:
        # Fill missing values
        df_copy['date_added'] = df_copy['date_added'].fillna('')
        
        # Only process rows with non-empty values
        # Check if date_added is a DataFrame (which would cause the error)
        if isinstance(df_copy['date_added'], pd.DataFrame):
            # If it's a DataFrame, convert to Series first
            df_copy['date_added'] = df_copy['date_added'].iloc[:, 0]
        
        # Then apply string operations
        mask = df_copy['date_added'].astype(str).str.len() > 0
        if mask.any():
            # Convert to datetime for valid values
            try:
                date_series = pd.to_datetime(df_copy.loc[mask, 'date_added'], errors='coerce')
                # Only extract info from valid dates
                valid_mask = ~date_series.isna()
                if valid_mask.any():
                    # Create year_added and month_added columns
                    if 'year_added' not in df_copy.columns:
                        df_copy['year_added'] = pd.NA
                    if 'month_added' not in df_copy.columns:
                        df_copy['month_added'] = pd.NA
                    
                    df_copy.loc[mask & valid_mask, 'year_added'] = date_series[valid_mask].dt.year
                    df_copy.loc[mask & valid_mask, 'month_added'] = date_series[valid_mask].dt.month
            except Exception as e:
                print(f"Error processing date_added column: {e}")
    else:
        # Create empty columns
        df_copy['date_added'] = ''
        df_copy['year_added'] = pd.NA
        df_copy['month_added'] = pd.NA
    
    # Process release_year column
    if 'release_year' in df_copy.columns:
        # Fill missing values
        df_copy['release_year'] = df_copy['release_year'].fillna('')
        
        # Check if release_year is a DataFrame (which would cause the error)
        if isinstance(df_copy['release_year'], pd.DataFrame):
            # If it's a DataFrame, convert to Series first
            df_copy['release_year'] = df_copy['release_year'].iloc[:, 0]
        
        # Only process rows with non-empty values
        mask = df_copy['release_year'].astype(str).str.len() > 0
        if mask.any():
            try:
                # Convert to numeric
                year_series = pd.to_numeric(df_copy.loc[mask, 'release_year'], errors='coerce')
                
                # Only process valid years
                valid_mask = ~year_series.isna()
                if valid_mask.any():
                    # Create decade column
                    if 'decade' not in df_copy.columns:
                        df_copy['decade'] = pd.NA
                    
                    df_copy.loc[mask & valid_mask, 'decade'] = (year_series[valid_mask] // 10) * 10
            except Exception as e:
                print(f"Error processing release_year column: {e}")
    else:
        # Create empty columns
        import datetime
        current_year = datetime.datetime.now().year
        df_copy['release_year'] = current_year  # Default to current year
        df_copy['decade'] = (current_year // 10) * 10
    
    return df_copy

def preprocess_data(df):
    """
    Apply all preprocessing steps to the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataset
    """
    try:
        # Create an empty DataFrame if input is None
        if df is None or df.empty:
            df = pd.DataFrame()
        
        # Ensure required columns exist
        required_columns = ['type', 'release_year']
        for col in required_columns:
            if col not in df.columns:
                if col == 'type':
                    df[col] = 'Unknown'
                elif col == 'release_year':
                    import datetime
                    df[col] = datetime.datetime.now().year
        
        # Apply preprocessing functions with error handling
        df = clean_duration(df)
        df = standardize_ratings(df)
        df = clean_text_columns(df)
        df = extract_release_date_info(df)
        
        return df
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Return the original dataframe if preprocessing fails
        return df 