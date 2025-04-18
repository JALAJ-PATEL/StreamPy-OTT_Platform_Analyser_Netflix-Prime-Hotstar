import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set color schemes
NETFLIX_COLOR = '#e50914'
DISNEY_COLOR = '#113CCF'
AMAZON_COLOR = '#00A8E1'

PLATFORM_COLORS = {
    'Netflix': NETFLIX_COLOR,
    'Disney+': DISNEY_COLOR,
    'Amazon Prime': AMAZON_COLOR
}

def plot_content_distribution(df):
    """
    Create a pie chart showing content distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a platform column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Pie chart figure
    """
    platform_counts = df['platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Count']
    
    fig = px.pie(platform_counts, values='Count', names='Platform', 
                title='Content Distribution by Platform',
                color='Platform',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_type_distribution_by_platform(df):
    """
    Create a bar chart showing content type distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform and type columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    type_platform = pd.crosstab(df['platform'], df['type'])
    
    fig = px.bar(type_platform, 
                title='Content Type Distribution by Platform',
                barmode='group',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Platform',
        yaxis_title='Count',
        legend_title_text='Content Type',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_ratings_distribution(df):
    """
    Create a bar chart showing ratings distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform and rating columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    # Check if rating column exists
    if 'rating' not in df.columns:
        return None
    
    ratings_counts = df.groupby(['platform', 'rating']).size().reset_index(name='count')
    
    fig = px.bar(ratings_counts, x='rating', y='count', color='platform', 
                title='Ratings Distribution by Platform',
                barmode='group',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Rating',
        yaxis_title='Count',
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_movie_duration_distribution(df):
    """
    Create a box plot showing movie duration distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform, type, duration_value, and duration_unit columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Box plot figure
    """
    # Check if required columns exist
    if not all(col in df.columns for col in ['type', 'duration_value', 'duration_unit']):
        return None
    
    # Filter for movies with minutes
    # Make a copy to avoid modifying the original dataframe
    df_movies = df.copy()
    
    # Check if duration_unit is a DataFrame
    if isinstance(df_movies['duration_unit'], pd.DataFrame):
        df_movies['duration_unit'] = df_movies['duration_unit'].iloc[:, 0]
    
    movies_duration = df_movies[(df_movies['type'] == 'Movie') & 
                      (df_movies['duration_unit'].str.contains('min', na=False))]
    
    if movies_duration.empty:
        return None
    
    fig = px.box(movies_duration, x='platform', y='duration_value', 
                title='Movie Duration Distribution (minutes)',
                color='platform',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Platform',
        yaxis_title='Duration (minutes)',
        legend_title_text='Platform',
        showlegend=False
    )
    
    return fig

def plot_tv_seasons_distribution(df):
    """
    Create a histogram showing TV show seasons distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform, type, duration_value, and duration_unit columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Histogram figure
    """
    # Check if required columns exist
    if not all(col in df.columns for col in ['type', 'duration_value', 'duration_unit']):
        return None
    
    # Filter for TV shows with seasons
    # Make a copy to avoid modifying the original dataframe
    df_tv = df.copy()
    
    # Check if duration_unit is a DataFrame
    if isinstance(df_tv['duration_unit'], pd.DataFrame):
        df_tv['duration_unit'] = df_tv['duration_unit'].iloc[:, 0]
    
    tv_seasons = df_tv[(df_tv['type'] == 'TV Show') & 
                      (df_tv['duration_unit'].str.contains('Season', na=False))]
    
    if tv_seasons.empty:
        return None
    
    fig = px.histogram(tv_seasons, x='duration_value', color='platform',
                     title='TV Shows by Number of Seasons',
                     barmode='group',
                     color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Number of Seasons',
        yaxis_title='Count',
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_release_year_trend(df):
    """
    Create a line chart showing content release trends by year and platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform and release_year columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Line chart figure
    """
    # Check if release_year column exists
    if 'release_year' not in df.columns:
        return None
    
    year_counts = df.groupby(['release_year', 'platform']).size().reset_index(name='count')
    
    fig = px.line(year_counts, x='release_year', y='count', color='platform',
                title='Content Release Trends by Year',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Release Year',
        yaxis_title='Count',
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_addition_trend(df):
    """
    Create a line chart showing content addition trends over time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform, date_added, year_added, and month_added columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Line chart figure
    """
    # Check if required columns exist
    if 'date_added' not in df.columns:
        return None
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert date_added to datetime if not already
    df_copy['date_added'] = pd.to_datetime(df_copy['date_added'], errors='coerce')
    
    # Filter out rows with invalid dates
    df_copy = df_copy.dropna(subset=['date_added'])
    
    # If no valid dates, return None
    if df_copy.empty:
        return None
    
    # Extract year and month
    df_copy['year_added'] = df_copy['date_added'].dt.year
    df_copy['month_added'] = df_copy['date_added'].dt.month
    
    # Group by year, month, and platform
    addition_trends = df_copy.groupby(['year_added', 'month_added', 'platform']).size().reset_index(name='count')
    
    # Create a date column for better visualization
    # Make sure both year and month are available and valid
    if 'year_added' in addition_trends.columns and 'month_added' in addition_trends.columns:
        # Filter out rows with missing year or month
        valid_dates = addition_trends.dropna(subset=['year_added', 'month_added'])
        if not valid_dates.empty:
            # Add day column and convert to datetime
            valid_dates['date'] = pd.to_datetime(
                dict(
                    year=valid_dates['year_added'], 
                    month=valid_dates['month_added'], 
                    day=1
                )
            )
            # Use only the valid dates for plotting
            addition_trends = valid_dates
            
            # Sort by date for consistent trend lines
            addition_trends = addition_trends.sort_values('date')
        else:
            # No valid dates to plot
            return None
    else:
        # Missing required columns
        return None
    
    fig = px.line(addition_trends, x='date', y='count', color='platform',
                title='Content Addition Trends Over Time',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Date Added',
        yaxis_title='Count',
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_top_genres(df, n=15):
    """
    Create a bar chart showing top genres
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a listed_in column with genres
    n : int, default=15
        Number of top genres to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    # Check if listed_in column exists
    if 'listed_in' not in df.columns:
        return None
    
    # Process genres
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    
    genre_counts = pd.DataFrame(all_genres, columns=['genre']).value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    genre_counts = genre_counts.sort_values('count', ascending=False).head(n)
    
    fig = px.bar(genre_counts, x='genre', y='count',
                title=f'Top {n} Genres Overall',
                color='count',
                color_continuous_scale=px.colors.sequential.Reds)
    
    fig.update_layout(
        xaxis_title='Genre',
        xaxis={'categoryorder':'total descending'},
        yaxis_title='Count'
    )
    
    return fig

def plot_genres_by_platform(df, n=10):
    """
    Create a bar chart showing genre distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform and listed_in columns
    n : int, default=10
        Number of top genres per platform to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    # Check if required columns exist
    if 'listed_in' not in df.columns or 'platform' not in df.columns:
        return None
    
    # Function to extract genres for each platform
    def get_platform_genres(platform_df):
        platform_genres = []
        for genres in platform_df['listed_in'].dropna():
            platform_genres.extend([g.strip() for g in genres.split(',')])
        return platform_genres
    
    # Get top genres for each platform
    platform_genre_data = []
    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform]
        genres = get_platform_genres(platform_df)
        
        if not genres:
            continue
            
        genre_df = pd.DataFrame(genres, columns=['genre']).value_counts().reset_index()
        genre_df.columns = ['genre', 'count']
        genre_df['platform'] = platform
        platform_genre_data.append(genre_df.head(n))
    
    if not platform_genre_data:
        return None
        
    combined_genres = pd.concat(platform_genre_data)
    
    fig = px.bar(combined_genres, x='genre', y='count', color='platform',
                title=f'Top {n} Genres by Platform',
                barmode='group',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Genre',
        xaxis={'categoryorder':'total descending'},
        yaxis_title='Count',
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_top_countries(df, n=15):
    """
    Create a bar chart showing top content producing countries
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a country column
    n : int, default=15
        Number of top countries to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    # Check if country column exists
    if 'country' not in df.columns:
        return None
    
    # Process countries
    all_countries = []
    for countries in df['country'].dropna():
        all_countries.extend([c.strip() for c in countries.split(',')])
    
    country_counts = pd.DataFrame(all_countries, columns=['country']).value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_counts = country_counts.sort_values('count', ascending=False).head(n)
    
    fig = px.bar(country_counts, x='country', y='count',
                title=f'Top {n} Content Producing Countries',
                color='count',
                color_continuous_scale=px.colors.sequential.Reds)
    
    fig.update_layout(
        xaxis_title='Country',
        xaxis={'categoryorder':'total descending'},
        yaxis_title='Count'
    )
    
    return fig

def plot_countries_by_platform(df, n=10):
    """
    Create a bar chart showing country distribution by platform
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing platform and country columns
    n : int, default=10
        Number of top countries per platform to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    # Check if required columns exist
    if 'country' not in df.columns or 'platform' not in df.columns:
        return None
    
    # Function to extract countries for each platform
    def get_platform_countries(platform_df):
        platform_countries = []
        for countries in platform_df['country'].dropna():
            platform_countries.extend([c.strip() for c in countries.split(',')])
        return platform_countries
    
    # Get top countries for each platform
    platform_country_data = []
    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform]
        countries = get_platform_countries(platform_df)
        
        if not countries:
            continue
            
        country_df = pd.DataFrame(countries, columns=['country']).value_counts().reset_index()
        country_df.columns = ['country', 'count']
        country_df['platform'] = platform
        platform_country_data.append(country_df.head(n))
    
    if not platform_country_data:
        return None
        
    combined_countries = pd.concat(platform_country_data)
    
    fig = px.bar(combined_countries, x='country', y='count', color='platform',
                title=f'Top {n} Production Countries by Platform',
                barmode='group',
                color_discrete_map=PLATFORM_COLORS)
    
    fig.update_layout(
        xaxis_title='Country',
        xaxis={'categoryorder':'total descending'},
        yaxis_title='Count',
        legend_title_text='Platform',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_world_map(df):
    """
    Create a choropleth map showing global content distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing a country column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Choropleth map figure
    """
    # Check if country column exists
    if 'country' not in df.columns:
        return None
    
    # Process countries
    all_countries = []
    for countries in df['country'].dropna():
        all_countries.extend([c.strip() for c in countries.split(',')])
    
    country_counts = pd.DataFrame(all_countries, columns=['country']).value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    if country_counts.empty:
        return None
    
    fig = px.choropleth(country_counts, 
                       locations='country', 
                       locationmode='country names',
                       color='count',
                       title='Content Production by Country',
                       color_continuous_scale=px.colors.sequential.Reds)
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    
    return fig 