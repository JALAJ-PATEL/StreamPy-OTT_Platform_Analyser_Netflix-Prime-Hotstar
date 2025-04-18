import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Import utility modules
from utils import data_loader, visualization, recommendation

# Set page configuration
st.set_page_config(
    page_title="Streaming Platform Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #e50914;
        margin-bottom: 1rem;
    }
    .platform-section {
        background-color: #f5f5f1;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .netflix {
        background-color: #e50914;
        color: white;
    }
    .disney {
        background-color: #113CCF;
        color: white;
    }
    .amazon {
        background-color: #00A8E1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='main-header'>Streaming Platform Content Analysis</h1>", unsafe_allow_html=True)
st.markdown("Analyze and compare content from Netflix, Disney+, and Amazon Prime Video")

# Helper function to verify data path exists
def verify_data_paths():
    data_dir = os.path.join("data")
    if not os.path.exists(data_dir):
        st.error(f"Data directory not found: {data_dir}")
        st.info("Please create a 'data' directory and place the dataset files there.")
        return False
    
    netflix_path = os.path.join(data_dir, "netflix_titles.csv")
    disney_path = os.path.join(data_dir, "disney_plus_titles.csv")
    amazon_path = os.path.join(data_dir, "amazon_prime_titles.csv")
    
    missing_files = []
    if not os.path.exists(netflix_path):
        missing_files.append("netflix_titles.csv")
    if not os.path.exists(disney_path):
        missing_files.append("disney_plus_titles.csv")
    if not os.path.exists(amazon_path):
        missing_files.append("amazon_prime_titles.csv")
    
    if missing_files:
        st.error(f"Missing data files: {', '.join(missing_files)}")
        st.info("""
        Please ensure the following files are in the 'data' directory:
        - netflix_titles.csv
        - disney_plus_titles.csv
        - amazon_prime_titles.csv
        
        The app needs at least one of these files to function properly.
        """)
        return False
    
    return True

# Try loading the data and handle any errors
try:
    # Verify data paths first
    if verify_data_paths():
        netflix_df, disney_df, amazon_df = data_loader.load_all_data()
        combined_df = data_loader.combine_data(netflix_df, disney_df, amazon_df)
        
        # Check if the combined dataframe has data
        if combined_df.empty:
            st.error("No data available. All datasets are empty.")
            data_loaded = False
        else:
            # Verify that required columns exist
            required_columns = ['platform', 'type', 'release_year']
            missing_columns = [col for col in required_columns if col not in combined_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns in data: {', '.join(missing_columns)}")
                data_loaded = False
            else:
                data_loaded = True
    else:
        data_loaded = False
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# Sidebar Navigation
if data_loaded:
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Section",
        ["Overview", "Content Comparison", "Release Trends", 
         "Genre Analysis", "Country Analysis", "Content Recommendation"]
    )
    
    # Platform selector
    platforms = st.sidebar.multiselect(
        "Select Platforms to Analyze",
        ["Netflix", "Disney+", "Amazon Prime"],
        default=["Netflix", "Disney+", "Amazon Prime"]
    )
    
    # Filter data based on selected platforms
    platform_mapping = {
        "Netflix": netflix_df,
        "Disney+": disney_df,
        "Amazon Prime": amazon_df
    }
    
    # Apply filters
    if platforms:
        filtered_df = data_loader.filter_data(combined_df, platforms=platforms)
    else:
        st.warning("Please select at least one platform")
        filtered_df = pd.DataFrame()
    
    # Additional filters
    content_type = st.sidebar.radio("Content Type", ["All", "Movie", "TV Show"])
    if content_type != "All":
        filtered_df = data_loader.filter_data(filtered_df, content_type=content_type)
    
    # Year range filter
    if not filtered_df.empty:
        min_year = int(filtered_df['release_year'].min())
        max_year = int(filtered_df['release_year'].max())
        year_range = st.sidebar.slider(
            "Release Year Range",
            min_year, max_year, (min_year, max_year)
        )
        filtered_df = data_loader.filter_data(filtered_df, year_range=year_range)
    
    # Overview Section
    if app_mode == "Overview":
        st.markdown("<h2 class='sub-header'>Platform Overview</h2>", unsafe_allow_html=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Titles", len(filtered_df))
        with col2:
            st.metric("Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
        with col3:
            st.metric("TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))
        
        # Platform content distribution
        st.subheader("Content Distribution by Platform")
        fig = visualization.plot_content_distribution(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Content type distribution
        st.subheader("Content Type Distribution by Platform")
        fig = visualization.plot_type_distribution_by_platform(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent additions
        st.subheader("Recent Additions")
        if 'date_added' in filtered_df.columns:
            # Convert date_added to datetime once and cache the result
            filtered_df_dates = filtered_df.copy()
            filtered_df_dates['date_added'] = pd.to_datetime(filtered_df_dates['date_added'], errors='coerce')
            
            # Format the date_added column as string to prevent jittering
            filtered_df_dates['date_added'] = filtered_df_dates['date_added'].dt.strftime('%Y-%m-%d')
            
            # Get recent additions - only use valid dates
            recent = filtered_df_dates.dropna(subset=['date_added']).sort_values('date_added', ascending=False).head(10)
            
            # Select columns and display
            if not recent.empty:
                st.dataframe(recent[['title', 'platform', 'type', 'date_added']], use_container_width=True)
            else:
                st.info("No recent additions data available.")
        else:
            st.info("Date information not available.")
    
    # Content Comparison Section
    elif app_mode == "Content Comparison":
        st.markdown("<h2 class='sub-header'>Content Comparison Across Platforms</h2>", unsafe_allow_html=True)
        
        # Ratings distribution
        st.subheader("Ratings Distribution")
        fig = visualization.plot_ratings_distribution(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rating information not available for comparison.")
        
        # Check for duration data availability first
        movie_fig = visualization.plot_movie_duration_distribution(filtered_df)
        tv_fig = visualization.plot_tv_seasons_distribution(filtered_df)
        
        # Only show Duration Analysis section if there's actual data
        if movie_fig is not None or tv_fig is not None:
            st.subheader("Duration Analysis")
            
            # Movie duration distribution
            if movie_fig:
                st.plotly_chart(movie_fig, use_container_width=True)
            
            # TV Shows seasons analysis
            if tv_fig:
                st.plotly_chart(tv_fig, use_container_width=True)
    
    # Release Trends Section
    elif app_mode == "Release Trends":
        st.markdown("<h2 class='sub-header'>Content Release Trends</h2>", unsafe_allow_html=True)
        
        # Content by release year
        st.subheader("Content by Release Year")
        fig = visualization.plot_release_year_trend(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Release year information not available for trend analysis.")
        
        # Content addition trends
        st.subheader("Content Addition Trends")
        fig = visualization.plot_addition_trend(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date added information not available for trend analysis.")
    
    # Genre Analysis Section
    elif app_mode == "Genre Analysis":
        st.markdown("<h2 class='sub-header'>Genre Analysis</h2>", unsafe_allow_html=True)
        
        # Top genres overall
        st.subheader("Top Genres Overall")
        fig = visualization.plot_top_genres(filtered_df, n=15)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Genre information not available for analysis.")
        
        # Genre distribution by platform
        if len(platforms) > 1:  # Only show if multiple platforms selected
            st.subheader("Genre Distribution by Platform")
            fig = visualization.plot_genres_by_platform(filtered_df, n=10)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Genre exploration
        st.subheader("Explore Content by Genre")
        
        # Get all unique genres
        all_genres = data_loader.extract_genres(filtered_df)
        
        if all_genres:
            selected_genre = st.selectbox("Select a Genre", sorted(all_genres))
            
            if selected_genre:
                # Get recommendations for the selected genre
                genre_titles = recommendation.get_genre_recommendations(
                    filtered_df, 
                    selected_genre, 
                    n=10, 
                    content_type=None if content_type == "All" else content_type
                )
                
                if not genre_titles.empty:
                    st.write(f"Top titles in the {selected_genre} genre:")
                    st.dataframe(
                        genre_titles[['title', 'platform', 'type', 'release_year', 'listed_in']],
                        use_container_width=True
                    )
                else:
                    st.info(f"No titles found for the {selected_genre} genre.")
        else:
            st.info("Genre information not available for exploration.")
    
    # Country Analysis Section
    elif app_mode == "Country Analysis":
        st.markdown("<h2 class='sub-header'>Content by Country</h2>", unsafe_allow_html=True)
        
        # Top countries overall
        st.subheader("Top Content Producing Countries")
        fig = visualization.plot_top_countries(filtered_df, n=15)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Country information not available for analysis.")
        
        # Country distribution by platform
        if len(platforms) > 1:  # Only show if multiple platforms selected
            st.subheader("Production Countries by Platform")
            fig = visualization.plot_countries_by_platform(filtered_df, n=10)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # World map visualization
        st.subheader("Global Content Distribution")
        fig = visualization.plot_world_map(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Country exploration
        st.subheader("Explore Content by Country")
        
        # Get all unique countries
        all_countries = data_loader.extract_countries(filtered_df)
        
        if all_countries:
            selected_country = st.selectbox("Select a Country", sorted(all_countries))
            
            if selected_country:
                # Filter titles from the selected country
                # Check if country is potentially a DataFrame
                country_df = filtered_df.copy()
                if isinstance(country_df['country'], pd.DataFrame):
                    country_df['country'] = country_df['country'].iloc[:, 0]
                
                # Now perform the filtering
                country_titles = country_df[country_df['country'].str.contains(selected_country, case=False, na=False)]
                
                if not country_titles.empty:
                    st.write(f"Content produced in {selected_country}:")
                    st.dataframe(
                        country_titles[['title', 'platform', 'type', 'release_year', 'listed_in']].head(20),
                        use_container_width=True
                    )
                else:
                    st.info(f"No titles found for {selected_country}.")
        else:
            st.info("Country information not available for exploration.")
    
    # Content Recommendation Section
    elif app_mode == "Content Recommendation":
        st.markdown("<h2 class='sub-header'>Content Recommendation</h2>", unsafe_allow_html=True)
        
        # Create tabs for different recommendation approaches
        tabs = st.tabs(["Title Search", "Genre-Based", "Advanced Options"])
        
        with tabs[0]:  # Title Search tab
            st.subheader("Find Similar Content")
            search_query = st.text_input("Search for a movie or TV show:")
            
            if search_query:
                # Search across all platforms
                search_results = data_loader.search_titles(filtered_df, search_query)
                
                if len(search_results) > 0:
                    st.write(f"Found {len(search_results)} matching titles:")
                    st.dataframe(search_results[['title', 'type', 'platform', 'release_year', 'rating']], 
                               use_container_width=True)
                    
                    # Allow selection for recommendations
                    selected_title = st.selectbox(
                        "Select a title to get recommendations:",
                        search_results['title'].tolist()
                    )
                    
                    if selected_title:
                        # Get the selected title's information
                        selected_row = filtered_df[filtered_df['title'] == selected_title].iloc[0]
                        
                        # Display information about the selected title
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("Selected Title")
                            st.markdown(f"**Title:** {selected_row['title']}")
                            st.markdown(f"**Type:** {selected_row['type']}")
                            st.markdown(f"**Platform:** {selected_row['platform']}")
                            st.markdown(f"**Release Year:** {selected_row['release_year']}")
                            if 'rating' in selected_row and not pd.isna(selected_row['rating']):
                                st.markdown(f"**Rating:** {selected_row['rating']}")
                        
                        with col2:
                            st.subheader("Description")
                            if 'description' in selected_row and not pd.isna(selected_row['description']):
                                st.write(selected_row['description'])
                            
                            if 'listed_in' in selected_row and not pd.isna(selected_row['listed_in']):
                                st.markdown(f"**Genres:** {selected_row['listed_in']}")
                            
                            if 'cast' in selected_row and not pd.isna(selected_row['cast']):
                                st.markdown(f"**Cast:** {selected_row['cast']}")
                            
                            if 'director' in selected_row and not pd.isna(selected_row['director']):
                                st.markdown(f"**Director:** {selected_row['director']}")
                        
                        # Generate recommendations
                        st.subheader("Similar Titles You Might Enjoy")
                        
                        recommendation_type = st.radio(
                            "Recommendation Method:",
                            ["Content-Based", "Hybrid"],
                            horizontal=True
                        )
                        
                        if recommendation_type == "Content-Based":
                            similar_titles = recommendation.get_content_based_recommendations(
                                filtered_df, selected_title, n=10
                            )
                        else:  # Hybrid
                            similar_titles = recommendation.get_hybrid_recommendations(
                                filtered_df, selected_title, n=10
                            )
                        
                        if not similar_titles.empty:
                            st.dataframe(similar_titles, use_container_width=True)
                        else:
                            st.info("Not enough similar content available for recommendations.")
                else:
                    st.write("No matching titles found. Try a different search term.")
            else:
                st.write("Enter a movie or TV show title to search and get recommendations.")
        
        with tabs[1]:  # Genre-Based tab
            st.subheader("Find Content by Genre")
            
            # Get all unique genres
            all_genres = data_loader.extract_genres(filtered_df)
            
            if all_genres:
                # Allow selecting multiple genres
                selected_genres = st.multiselect(
                    "Select Genres",
                    options=sorted(all_genres),
                    default=None
                )
                
                if selected_genres:
                    # Filter by platform
                    platform_filter = st.selectbox(
                        "Platform",
                        ["All"] + sorted(filtered_df['platform'].unique().tolist())
                    )
                    
                    platform = None if platform_filter == "All" else platform_filter
                    
                    # Get recommendations based on selected genres
                    genre_recommendations = recommendation.get_genre_recommendations(
                        filtered_df,
                        selected_genres,
                        n=15,
                        content_type=None if content_type == "All" else content_type,
                        platform=platform
                    )
                    
                    if not genre_recommendations.empty:
                        st.write(f"Top titles matching your genre selection:")
                        st.dataframe(
                            genre_recommendations[['title', 'platform', 'type', 'release_year', 'listed_in']],
                            use_container_width=True
                        )
                    else:
                        st.info("No titles found matching your genre selection.")
            else:
                st.info("Genre information not available for recommendations.")
        
        with tabs[2]:  # Advanced Options tab
            st.subheader("Advanced Recommendation Options")
            
            st.write("Customize your recommendation experience:")
            
            # Content-Based vs. Genre-Based weight slider
            st.write("For hybrid recommendations, adjust the weight of each approach:")
            content_weight = st.slider(
                "Content-Based Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7,
                step=0.1
            )
            genre_weight = 1.0 - content_weight
            
            st.write(f"Current weights: {content_weight:.1f} Content-Based, {genre_weight:.1f} Genre-Based")
            
            # Custom Recommendation Demo
            st.subheader("Try Custom Recommendations")
            
            search_query = st.text_input("Search for a title:", key="adv_search")
            
            if search_query:
                # Search across all platforms
                search_results = data_loader.search_titles(filtered_df, search_query)
                
                if len(search_results) > 0:
                    selected_title = st.selectbox(
                        "Select a title:",
                        search_results['title'].tolist(),
                        key="adv_select"
                    )
                    
                    if selected_title:
                        hybrid_titles = recommendation.get_hybrid_recommendations(
                            filtered_df, 
                            selected_title, 
                            weight_content=content_weight,
                            weight_genre=genre_weight,
                            n=10
                        )
                        
                        if not hybrid_titles.empty:
                            st.write("Custom Hybrid Recommendations:")
                            st.dataframe(hybrid_titles, use_container_width=True)
                        else:
                            st.info("Not enough data for custom recommendations.")
                else:
                    st.write("No matching titles found. Try a different search term.")

else:
    st.error("Failed to load data. Please check your data files and try again.")

# Footer
st.markdown("---")
st.markdown("Created with Streamlit • Data sourced from Netflix, Disney+, and Amazon Prime catalogs") 