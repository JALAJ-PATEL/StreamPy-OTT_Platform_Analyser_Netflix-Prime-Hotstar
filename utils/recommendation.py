import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_soup(x):
    """
    Create a text 'soup' by combining various features for content-based filtering
    
    Parameters:
    -----------
    x : pandas.Series
        Row of a DataFrame containing content information
        
    Returns:
    --------
    str
        Combined text features
    """
    features = []
    
    # Add genres
    if 'listed_in' in x and not pd.isna(x['listed_in']):
        features.append(x['listed_in'])
    
    # Add director
    if 'director' in x and not pd.isna(x['director']):
        features.append(x['director'])
    
    # Add cast
    if 'cast' in x and not pd.isna(x['cast']):
        features.append(x['cast'])
    
    # Add description
    if 'description' in x and not pd.isna(x['description']):
        features.append(x['description'])
    
    return ' '.join(features)

def get_content_based_recommendations(df, title, n=10, content_type=None):
    """
    Generate content-based recommendations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing content information
    title : str
        Title to get recommendations for
    n : int, default=10
        Number of recommendations to return
    content_type : str, default=None
        Filter by content type (Movie or TV Show)
        
    Returns:
    --------
    pandas.DataFrame
        Similar titles
    """
    # Check if title exists in the dataset
    if title not in df['title'].values:
        return pd.DataFrame()
    
    # Get the row of the selected title
    selected_row = df[df['title'] == title].iloc[0]
    
    # Filter by content type if specified, or use the same type as the selected title
    if content_type is None:
        content_type = selected_row['type']
    
    content_type_df = df[df['type'] == content_type].copy()
    
    # Ensure we have more than just the selected title
    if len(content_type_df) <= 1:
        return pd.DataFrame()
    
    # Create text soup for each title
    content_type_df['soup'] = content_type_df.apply(create_soup, axis=1)
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(content_type_df['soup'])
    
    # Compute similarity scores
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the selected title
    idx = content_type_df[content_type_df['title'] == title].index[0]
    
    # Get similarity scores and sort
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n similar titles (excluding itself)
    sim_scores = sim_scores[1:n+1]
    similar_indices = [i[0] for i in sim_scores]
    
    # Add similarity score to the output
    similar_titles = content_type_df.iloc[similar_indices].copy()
    similarity_scores = [i[1] for i in sim_scores]
    similar_titles['similarity_score'] = similarity_scores
    
    # Select columns for output
    output_columns = ['title', 'platform', 'type', 'release_year', 'similarity_score']
    
    # Include additional columns if they exist
    if 'rating' in similar_titles.columns:
        output_columns.append('rating')
    if 'duration' in similar_titles.columns:
        output_columns.append('duration')
    
    return similar_titles[output_columns]

def get_genre_recommendations(df, genres, n=10, content_type=None, platform=None):
    """
    Generate recommendations based on genres
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing content information
    genres : str or list
        Genre(s) to get recommendations for
    n : int, default=10
        Number of recommendations to return
    content_type : str, default=None
        Filter by content type (Movie or TV Show)
    platform : str, default=None
        Filter by platform
        
    Returns:
    --------
    pandas.DataFrame
        Recommended titles
    """
    # Filter by content type if specified
    if content_type is not None:
        filtered_df = df[df['type'] == content_type].copy()
    else:
        filtered_df = df.copy()
    
    # Filter by platform if specified
    if platform is not None:
        filtered_df = filtered_df[filtered_df['platform'] == platform]
    
    # Convert genres to list if it's a string
    if isinstance(genres, str):
        genres = [genres]
    
    # Filter titles containing any of the specified genres
    genre_matches = []
    for _, row in filtered_df.iterrows():
        if 'listed_in' in row and not pd.isna(row['listed_in']):
            title_genres = [g.strip() for g in row['listed_in'].split(',')]
            # Check if any of the requested genres is in title's genres
            if any(genre in title_genres for genre in genres):
                genre_matches.append(row)
    
    # Convert to DataFrame
    if not genre_matches:
        return pd.DataFrame()
    
    matches_df = pd.DataFrame(genre_matches)
    
    # Count how many of the requested genres each title matches
    matches_df['genre_match_count'] = matches_df.apply(
        lambda x: sum(genre in [g.strip() for g in x['listed_in'].split(',')] for genre in genres),
        axis=1
    )
    
    # Sort by match count and then by popularity proxies (like rating if available)
    if 'rating' in matches_df.columns:
        matches_df = matches_df.sort_values(['genre_match_count', 'rating'], ascending=[False, False])
    else:
        matches_df = matches_df.sort_values('genre_match_count', ascending=False)
    
    # Select top n recommendations
    top_recommendations = matches_df.head(n)
    
    # Select columns for output
    output_columns = ['title', 'platform', 'type', 'release_year', 'listed_in']
    
    # Include additional columns if they exist
    if 'rating' in top_recommendations.columns:
        output_columns.append('rating')
    if 'duration' in top_recommendations.columns:
        output_columns.append('duration')
    
    return top_recommendations[output_columns]

def get_hybrid_recommendations(df, title, weight_content=0.7, weight_genre=0.3, n=10):
    """
    Generate hybrid recommendations by combining content-based and genre-based approaches
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing content information
    title : str
        Title to get recommendations for
    weight_content : float, default=0.7
        Weight to assign to content-based recommendations
    weight_genre : float, default=0.3
        Weight to assign to genre-based recommendations
    n : int, default=10
        Number of recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        Hybrid recommendations
    """
    # Check if title exists in the dataset
    if title not in df['title'].values:
        return pd.DataFrame()
    
    # Get the row of the selected title
    selected_row = df[df['title'] == title].iloc[0]
    
    # Get content-based recommendations
    content_recs = get_content_based_recommendations(df, title, n=n*2)
    
    # If no content-based recommendations, return empty DataFrame
    if content_recs.empty:
        return pd.DataFrame()
    
    # Get genres of the selected title
    if 'listed_in' in selected_row and not pd.isna(selected_row['listed_in']):
        genres = [g.strip() for g in selected_row['listed_in'].split(',')]
        
        # Get genre-based recommendations
        genre_recs = get_genre_recommendations(
            df, 
            genres, 
            n=n*2, 
            content_type=selected_row['type']
        )
        
        # If genre recommendations exist, combine with content-based
        if not genre_recs.empty:
            # Normalize similarity scores in content_recs
            max_sim = content_recs['similarity_score'].max()
            min_sim = content_recs['similarity_score'].min()
            if max_sim > min_sim:
                content_recs['norm_score'] = (content_recs['similarity_score'] - min_sim) / (max_sim - min_sim)
            else:
                content_recs['norm_score'] = 1.0
            
            # Combine recommendations
            all_recs = pd.concat([
                content_recs[['title', 'norm_score']].rename(columns={'norm_score': 'content_score'}),
                genre_recs[['title', 'genre_match_count']]
            ], axis=0)
            
            # Group by title and aggregate scores
            all_recs = all_recs.groupby('title').agg({
                'content_score': 'max',
                'genre_match_count': 'max'
            }).reset_index()
            
            # Normalize genre match count
            max_genre = all_recs['genre_match_count'].max()
            if max_genre > 0:
                all_recs['norm_genre_score'] = all_recs['genre_match_count'] / max_genre
            else:
                all_recs['norm_genre_score'] = 0
            
            # Calculate hybrid score
            all_recs['hybrid_score'] = (
                weight_content * all_recs['content_score'].fillna(0) +
                weight_genre * all_recs['norm_genre_score'].fillna(0)
            )
            
            # Sort by hybrid score
            all_recs = all_recs.sort_values('hybrid_score', ascending=False)
            
            # Get top n recommendations
            top_recs = all_recs.head(n)
            
            # Merge with original dataset to get full information
            hybrid_recs = pd.merge(top_recs, df, on='title', how='left')
            
            # Select columns for output
            output_columns = ['title', 'platform', 'type', 'release_year', 'hybrid_score']
            
            # Include additional columns if they exist
            if 'rating' in hybrid_recs.columns:
                output_columns.append('rating')
            if 'duration' in hybrid_recs.columns:
                output_columns.append('duration')
            
            return hybrid_recs[output_columns]
    
    # If no genre recommendations, return content-based recommendations
    return content_recs.head(n) 