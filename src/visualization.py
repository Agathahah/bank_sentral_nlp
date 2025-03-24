# -*- coding: utf-8 -*-
"""
visualization.py - Visualization functions for Bank Sentral NLP project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_sentiment_time_series(data, date_col, sentiment_col, title="Sentiment Over Time", 
                              output_path=None, figsize=(12, 6)):
    """
    Plot sentiment time series
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing date and sentiment columns
    date_col : str
        Name of date column
    sentiment_col : str
        Name of sentiment column
    title : str, default "Sentiment Over Time"
        Plot title
    output_path : str, optional
        Path to save the plot
    figsize : tuple, default (12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure
    """
    plt.figure(figsize=figsize)
    
    # Ensure date column is datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort by date
    data = data.sort_values(date_col)
    
    # Plot
    plt.plot(data[date_col], data[sentiment_col], 'o-', color='blue')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add trendline
    z = np.polyfit(data.index, data[sentiment_col], 1)
    p = np.poly1d(z)
    plt.plot(data[date_col], p(data.index), "r--", alpha=0.7)
    
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return plt.gcf()

def plot_sentiment_distribution(data, sentiment_col, category_col=None, title="Sentiment Distribution", 
                               output_path=None, figsize=(10, 6)):
    """
    Plot sentiment distribution
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing sentiment column
    sentiment_col : str
        Name of sentiment column
    category_col : str, optional
        Name of category column for grouping
    title : str, default "Sentiment Distribution"
        Plot title
    output_path : str, optional
        Path to save the plot
    figsize : tuple, default (10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure
    """
    plt.figure(figsize=figsize)
    
    if category_col:
        # Group by category
        categories = data[category_col].unique()
        
        for i, category in enumerate(categories):
            category_data = data[data[category_col] == category]
            sns.kdeplot(category_data[sentiment_col], label=category, fill=True, alpha=0.3)
    else:
        # Overall distribution
        sns.histplot(data[sentiment_col], kde=True, bins=20)
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if category_col:
        plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return plt.gcf()

def create_sentiment_wordcloud(texts, sentiments, sentiment_type='positive', mask=None, 
                            output_path=None, width=800, height=400, 
                            background_color='white', max_words=200):
    """
    Create wordcloud from texts based on sentiment
    
    Parameters:
    -----------
    texts : list
        List of texts
    sentiments : list
        List of sentiment labels or scores
    sentiment_type : str, default 'positive'
        Type of sentiment to include ('positive', 'negative', 'neutral', or 'all')
    mask : numpy.ndarray, optional
        Mask image for wordcloud
    output_path : str, optional
        Path to save the wordcloud
    width : int, default 800
        Wordcloud width
    height : int, default 400
        Wordcloud height
    background_color : str, default 'white'
        Background color
    max_words : int, default 200
        Maximum number of words
        
    Returns:
    --------
    WordCloud
        The wordcloud object
    """
    # Filter texts based on sentiment
    if sentiment_type == 'positive':
        filtered_texts = [text for text, sent in zip(texts, sentiments) if sent > 0.1]
    elif sentiment_type == 'negative':
        filtered_texts = [text for text, sent in zip(texts, sentiments) if sent < -0.1]
    elif sentiment_type == 'neutral':
        filtered_texts = [text for text, sent in zip(texts, sentiments) if -0.1 <= sent <= 0.1]
    else:
        filtered_texts = texts
    
    if not filtered_texts:
        print(f"No texts found for sentiment type: {sentiment_type}")
        return None
    
    # Combine texts
    text = ' '.join(filtered_texts)
    
    # Create wordcloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        mask=mask,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"{sentiment_type.capitalize()} Sentiment Wordcloud")
    plt.tight_layout(pad=0)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wordcloud saved to {output_path}")
    
    return wordcloud

def visualize_embeddings(embeddings, labels, method='tsne', n_components=2, perplexity=30,
                      output_path=None, figsize=(10, 8), title="Embeddings Visualization"):
    """
    Visualize embeddings using dimensionality reduction
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Embeddings matrix
    labels : list
        Labels for coloring points
    method : str, default 'tsne'
        Dimensionality reduction method ('tsne' or 'pca')
    n_components : int, default 2
        Number of components for dimensionality reduction
    perplexity : int, default 30
        Perplexity parameter for t-SNE
    output_path : str, optional
        Path to save the plot
    figsize : tuple, default (10, 8)
        Figure size
    title : str, default "Embeddings Visualization"
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure
    """
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    vis_df = pd.DataFrame(reduced_embeddings, columns=[f'Component {i+1}' for i in range(n_components)])
    vis_df['Label'] = labels
    
    # Plot
    plt.figure(figsize=figsize)
    
    if n_components == 2:
        # 2D plot
        sns.scatterplot(
            x='Component 1',
            y='Component 2',
            hue='Label',
            palette='viridis',
            data=vis_df,
            alpha=0.7
        )
    elif n_components == 3:
        # 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        categories = vis_df['Label'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            cat_data = vis_df[vis_df['Label'] == category]
            ax.scatter(
                cat_data['Component 1'],
                cat_data['Component 2'],
                cat_data['Component 3'],
                c=[colors[i]],
                label=category,
                alpha=0.7
            )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
    
    plt.title(f"{title} ({method.upper()})")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return plt.gcf()

def create_interactive_dashboard(data, date_col, sentiment_col, category_col=None, title="Sentiment Dashboard"):
    """
    Create interactive dashboard using Plotly
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing date and sentiment columns
    date_col : str
        Name of date column
    sentiment_col : str
        Name of sentiment column
    category_col : str, optional
        Name of category column for grouping
    title : str, default "Sentiment Dashboard"
        Dashboard title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The figure
    """
    # Ensure date column is datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort by date
    data = data.sort_values(date_col)
    
    # Create figure
    fig = go.Figure()
    
    if category_col:
        # Group by category
        categories = data[category_col].unique()
        
        for category in categories:
            category_data = data[data[category_col] == category]
            fig.add_trace(go.Scatter(
                x=category_data[date_col],
                y=category_data[sentiment_col],
                mode='lines+markers',
                name=category
            ))
    else:
        # Overall trend
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=data[sentiment_col],
            mode='lines+markers',
            name='Sentiment'
        ))
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=data[date_col].min(),
        y0=0,
        x1=data[date_col].max(),
        y1=0,
        line=dict(color="gray", width=2, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        legend_title="Category" if category_col else None,
        hovermode="closest"
    )
    
    return fig