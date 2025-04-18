import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import os
from .preprocessing import preprocess_data

def load_netflix_data(filepath=os.path.join('data', 'netflix_titles.csv')):
    """
    Load and process Netflix dataset
    
    Parameters:
    -----------
    filepath : str, default=os.path.join('data', 'netflix_titles.csv')
        Path to the Netflix dataset
        
    Returns:
    --------
    pandas.DataFrame
        Processed Netflix dataset
    """
    try:
        # Try to load the file
        df = pd.read_csv(filepath)
        
        # Add platform column explicitly
        df['platform'] = 'Netflix'
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Verify platform column exists
        if 'platform' not in df.columns:
            df['platform'] = 'Netflix'
        
        return df
    except Exception as e:
        st.error(f"Error loading Netflix data: {e}")
        # Return empty DataFrame with platform column
        empty_df = pd.DataFrame()
        empty_df['platform'] = ['Netflix']
        return empty_df

def load_disney_data(filepath=os.path.join('data', 'disney_plus_titles.csv')):
    """
    Load and process Disney+ dataset
    
    Parameters:
    -----------
    filepath : str, default=os.path.join('data', 'disney_plus_titles.csv')
        Path to the Disney+ dataset
        
    Returns:
    --------
    pandas.DataFrame
        Processed Disney+ dataset
    """
    try:
        # Try to load the file
        df = pd.read_csv(filepath)
        
        # Add platform column explicitly
        df['platform'] = 'Disney+'
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Verify platform column exists
        if 'platform' not in df.columns:
            df['platform'] = 'Disney+'
        
        return df
    except Exception as e:
        st.error(f"Error loading Disney+ data: {e}")
        # Return empty DataFrame with platform column
        empty_df = pd.DataFrame()
        empty_df['platform'] = ['Disney+']
        return empty_df

def load_amazon_data(filepath=os.path.join('data', 'amazon_prime_titles.csv')):
    """
    Load and process Amazon Prime dataset
    
    Parameters:
    -----------
    filepath : str, default=os.path.join('data', 'amazon_prime_titles.csv')
        Path to the Amazon Prime dataset
        
    Returns:
    --------
    pandas.DataFrame
        Processed Amazon Prime dataset
    """
    try:
        # Try to load the file
        df = pd.read_csv(filepath)
        
        # Add platform column explicitly
        df['platform'] = 'Amazon Prime'
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Verify platform column exists
        if 'platform' not in df.columns:
            df['platform'] = 'Amazon Prime'
        
        return df
    except Exception as e:
        st.error(f"Error loading Amazon Prime data: {e}")
        # Return empty DataFrame with platform column
        empty_df = pd.DataFrame()
        empty_df['platform'] = ['Amazon Prime']
        return empty_df

@st.cache_data
def load_all_data():
    """
    Load and process all three datasets
    
    Returns:
    --------
    tuple
        Triple of processed DataFrames (netflix_df, disney_df, amazon_df)
    """
    try:
        netflix_df = load_netflix_data()
        # Ensure platform column exists
        if not netflix_df.empty and 'platform' not in netflix_df.columns:
            netflix_df['platform'] = 'Netflix'
    except Exception as e:
        st.error(f"Error loading Netflix data: {e}")
        netflix_df = pd.DataFrame({'platform': ['Netflix']})
    
    try:
        disney_df = load_disney_data()
        # Ensure platform column exists
        if not disney_df.empty and 'platform' not in disney_df.columns:
            disney_df['platform'] = 'Disney+'
    except Exception as e:
        st.error(f"Error loading Disney+ data: {e}")
        disney_df = pd.DataFrame({'platform': ['Disney+']})
    
    try:
        amazon_df = load_amazon_data()
        # Ensure platform column exists
        if not amazon_df.empty and 'platform' not in amazon_df.columns:
            amazon_df['platform'] = 'Amazon Prime'
    except Exception as e:
        st.error(f"Error loading Amazon Prime data: {e}")
        amazon_df = pd.DataFrame({'platform': ['Amazon Prime']})
    
    return netflix_df, disney_df, amazon_df

def combine_data(netflix_df, disney_df, amazon_df):
    """
    Combine all three datasets into a single DataFrame
    
    Parameters:
    -----------
    netflix_df : pandas.DataFrame
        Processed Netflix dataset
    disney_df : pandas.DataFrame
        Processed Disney+ dataset
    amazon_df : pandas.DataFrame
        Processed Amazon Prime dataset
        
    Returns:
    --------
    pandas.DataFrame
        Combined dataset
    """
    # Ensure platform column exists in each dataframe
    if netflix_df is not None and not netflix_df.empty and 'platform' not in netflix_df.columns:
        netflix_df['platform'] = 'Netflix'
        
    if disney_df is not None and not disney_df.empty and 'platform' not in disney_df.columns:
        disney_df['platform'] = 'Disney+'
        
    if amazon_df is not None and not amazon_df.empty and 'platform' not in amazon_df.columns:
        amazon_df['platform'] = 'Amazon Prime'
    
    # Create a list of valid dataframes
    dfs = []
    if netflix_df is not None and not netflix_df.empty:
        dfs.append(netflix_df)
    if disney_df is not None and not disney_df.empty:
        dfs.append(disney_df)
    if amazon_df is not None and not amazon_df.empty:
        dfs.append(amazon_df)
    
    # Combine the dataframes
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def filter_data(df, platforms=None, content_type=None, year_range=None):
    """
    Filter the dataset based on various criteria
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to filter
    platforms : list, default=None
        List of platforms to include
    content_type : str, default=None
        Type of content to include (Movie or TV Show)
    year_range : tuple, default=None
        Range of release years to include (min_year, max_year)
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataset
    """
    filtered_df = df.copy()
    
    # Check if platform column exists, if not, create it with a default value
    if 'platform' not in filtered_df.columns:
        st.warning("Column 'platform' not found in the dataset. Creating it with default value 'Unknown'.")
        filtered_df['platform'] = 'Unknown'
    
    # Filter by platform
    if platforms and 'platform' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['platform'].isin(platforms)]
    elif platforms:
        st.error("Column 'platform' not found in the dataset. Cannot filter by platform.")
    
    # Check if type column exists, if not, create it with a default value
    if 'type' not in filtered_df.columns:
        st.warning("Column 'type' not found in the dataset. Creating it with default value 'Unknown'.")
        filtered_df['type'] = 'Unknown'
        
    # Filter by content type
    if content_type and content_type != "All" and 'type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    elif content_type and content_type != "All":
        st.error("Column 'type' not found in the dataset. Cannot filter by content type.")
    
    # Check if release_year column exists, if not, create it with a default value
    if 'release_year' not in filtered_df.columns:
        st.warning("Column 'release_year' not found in the dataset. Creating it with current year.")
        import datetime
        filtered_df['release_year'] = datetime.datetime.now().year
        
    # Filter by release year range
    if year_range and 'release_year' in filtered_df.columns:
        min_year, max_year = year_range
        filtered_df = filtered_df[(filtered_df['release_year'] >= min_year) & 
                                (filtered_df['release_year'] <= max_year)]
    elif year_range:
        st.error("Column 'release_year' not found in the dataset. Cannot filter by year range.")
    
    return filtered_df

def extract_genres(df):
    """
    Extract all unique genres from the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a listed_in column with genres
        
    Returns:
    --------
    list
        List of unique genres
    """
    if 'listed_in' not in df.columns:
        return []
    
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    
    return list(set(all_genres))

def extract_countries(df):
    """
    Extract all unique countries from the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a country column
        
    Returns:
    --------
    list
        List of unique countries
    """
    if 'country' not in df.columns:
        return []
    
    all_countries = []
    for countries in df['country'].dropna():
        all_countries.extend([c.strip() for c in countries.split(',')])
    
    return list(set(all_countries))

def search_titles(df, query):
    """
    Search for titles in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to search in
    query : str
        Search query
        
    Returns:
    --------
    pandas.DataFrame
        Matching titles
    """
    # Check if title column exists
    if 'title' not in df.columns:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Check if title is a DataFrame (which would cause the error)
    if isinstance(df_copy['title'], pd.DataFrame):
        # If it's a DataFrame, convert to Series first
        df_copy['title'] = df_copy['title'].iloc[:, 0]
        
    # Now perform the search
    return df_copy[df_copy['title'].str.contains(query, case=False, na=False)] 