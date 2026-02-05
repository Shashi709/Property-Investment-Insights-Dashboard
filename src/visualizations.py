"""
Visualization Module for Property Investment Insights Dashboard
Author: Shashi Raj

This module provides:
- Interactive Plotly visualizations
- KPI metric cards
- Geospatial mapping components
- Advanced chart configurations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, Tuple


# Color scheme for consistent branding
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb33',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient': ['#667eea', '#764ba2'],
    'palette': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
}

# Crime index color mapping
CRIME_COLORS = {
    'Low': '#2ca02c',
    'Medium': '#ffbb33',
    'High': '#d62728'
}


def create_price_distribution_chart(
    df: pd.DataFrame,
    color_by: Optional[str] = None,
    nbins: int = 30
) -> go.Figure:
    """
    Create an interactive histogram of listing prices.

    Args:
        df: DataFrame with listing_price column
        color_by: Optional column to color by
        nbins: Number of histogram bins

    Returns:
        Plotly Figure object
    """
    fig = px.histogram(
        df,
        x='listing_price',
        nbins=nbins,
        color=color_by,
        title='📊 Listing Price Distribution',
        labels={'listing_price': 'Listing Price ($)', 'count': 'Number of Properties'},
        color_discrete_sequence=COLORS['palette'],
        marginal='box'
    )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        showlegend=True if color_by else False,
        xaxis_tickprefix='$',
        xaxis_tickformat=',.0f',
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_price_vs_sqft_scatter(
    df: pd.DataFrame,
    color_by: str = 'school_rating',
    size_by: Optional[str] = 'bedrooms',
    trendline: bool = True
) -> go.Figure:
    """
    Create a scatter plot of price vs square footage.

    Args:
        df: DataFrame with required columns
        color_by: Column to use for color encoding
        size_by: Optional column for size encoding
        trendline: Whether to add a trendline

    Returns:
        Plotly Figure object
    """
    # Prepare size values
    size_values = None
    if size_by and size_by in df.columns:
        size_values = df[size_by].fillna(1).clip(lower=1) * 5

    fig = px.scatter(
        df,
        x='sq_ft',
        y='listing_price',
        color=color_by if color_by in df.columns else None,
        size=size_values,
        hover_data=['raw_address', 'bedrooms', 'matched_zip', 'price_per_sqft'],
        title='🏠 Price vs. Square Footage',
        labels={
            'sq_ft': 'Square Footage',
            'listing_price': 'Listing Price ($)',
            'school_rating': 'School Rating',
            'bedrooms': 'Bedrooms'
        },
        color_continuous_scale='Viridis',
        trendline='ols' if trendline else None
    )

    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.0f',
        font=dict(family='Segoe UI, sans-serif')
    )

    fig.update_traces(
        marker=dict(opacity=0.7, line=dict(width=1, color='white'))
    )

    return fig


def create_price_per_sqft_by_zip(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart of average price per sqft by ZIP code.

    Args:
        df: DataFrame with matched_zip and price_per_sqft columns

    Returns:
        Plotly Figure object
    """
    # Aggregate by ZIP code
    zip_stats = df.groupby('matched_zip').agg({
        'price_per_sqft': 'mean',
        'listing_price': 'mean',
        'sq_ft': 'mean',
        'raw_address': 'count'
    }).reset_index()
    zip_stats.columns = ['ZIP Code', 'Avg Price/SqFt', 'Avg Price', 'Avg SqFt', 'Count']
    zip_stats = zip_stats.sort_values('Avg Price/SqFt', ascending=True)

    fig = px.bar(
        zip_stats,
        x='ZIP Code',
        y='Avg Price/SqFt',
        color='Avg Price/SqFt',
        text='Count',
        title='💰 Average Price per Sq.Ft by ZIP Code',
        labels={'Avg Price/SqFt': 'Price per Sq.Ft ($)'},
        color_continuous_scale='RdYlGn_r',
        hover_data=['Avg Price', 'Avg SqFt', 'Count']
    )

    fig.update_layout(
        template='plotly_white',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.0f',
        xaxis_tickangle=-45,
        font=dict(family='Segoe UI, sans-serif')
    )

    fig.update_traces(textposition='outside', textfont_size=10)

    return fig


def create_school_rating_vs_price(df: pd.DataFrame) -> go.Figure:
    """
    Create a bubble chart showing school rating vs listing price.

    Args:
        df: DataFrame with required columns

    Returns:
        Plotly Figure object
    """
    # Aggregate by ZIP code
    zip_data = df.groupby('matched_zip').agg({
        'school_rating': 'first',
        'listing_price': 'mean',
        'crime_index': 'first',
        'median_income': 'first',
        'raw_address': 'count'
    }).reset_index()
    zip_data.columns = ['ZIP', 'School Rating', 'Avg Price', 'Crime Level', 'Median Income', 'Listings']

    fig = px.scatter(
        zip_data,
        x='School Rating',
        y='Avg Price',
        size='Listings',
        color='Crime Level',
        color_discrete_map=CRIME_COLORS,
        hover_data=['ZIP', 'Median Income', 'Listings'],
        title='📚 School Rating vs. Average Listing Price',
        labels={'Avg Price': 'Average Listing Price ($)'}
    )

    fig.update_layout(
        template='plotly_white',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.0f',
        font=dict(family='Segoe UI, sans-serif')
    )

    fig.update_traces(
        marker=dict(opacity=0.8, line=dict(width=2, color='white'))
    )

    return fig


def create_investment_score_gauge(score: float) -> go.Figure:
    """
    Create a gauge chart for investment score.

    Args:
        score: Investment score (0-100)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Investment Score", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': COLORS['success']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLORS['primary']},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ]
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_crime_index_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart of crime index distribution.

    Args:
        df: DataFrame with crime_index column

    Returns:
        Plotly Figure object
    """
    crime_counts = df['crime_index'].value_counts().reset_index()
    crime_counts.columns = ['Crime Level', 'Count']

    fig = px.pie(
        crime_counts,
        values='Count',
        names='Crime Level',
        title='🔒 Crime Index Distribution',
        color='Crime Level',
        color_discrete_map=CRIME_COLORS,
        hole=0.4
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value'
    )

    fig.update_layout(
        template='plotly_white',
        showlegend=True,
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_income_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a violin plot of median income by crime level.

    Args:
        df: DataFrame with median_income and crime_index columns

    Returns:
        Plotly Figure object
    """
    fig = px.violin(
        df,
        x='crime_index',
        y='median_income',
        color='crime_index',
        color_discrete_map=CRIME_COLORS,
        box=True,
        points='all',
        title='💵 Median Income Distribution by Crime Level',
        labels={'median_income': 'Median Income ($)', 'crime_index': 'Crime Level'}
    )

    fig.update_layout(
        template='plotly_white',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.0f',
        showlegend=False,
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_bedrooms_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Create a grouped analysis of bedrooms vs price and sqft.

    Args:
        df: DataFrame with bedrooms, listing_price, sq_ft columns

    Returns:
        Plotly Figure object
    """
    bedroom_stats = df.groupby('bedrooms').agg({
        'listing_price': 'mean',
        'sq_ft': 'mean',
        'price_per_sqft': 'mean',
        'raw_address': 'count'
    }).reset_index()
    bedroom_stats.columns = ['Bedrooms', 'Avg Price', 'Avg SqFt', 'Avg Price/SqFt', 'Count']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Price by Bedrooms', 'Average Sq.Ft by Bedrooms'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    fig.add_trace(
        go.Bar(
            x=bedroom_stats['Bedrooms'],
            y=bedroom_stats['Avg Price'],
            name='Avg Price',
            marker_color=COLORS['primary'],
            text=bedroom_stats['Count'].apply(lambda x: f'{x} listings'),
            textposition='outside'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=bedroom_stats['Bedrooms'],
            y=bedroom_stats['Avg SqFt'],
            name='Avg SqFt',
            marker_color=COLORS['secondary']
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='🛏️ Bedroom Analysis',
        template='plotly_white',
        showlegend=False,
        font=dict(family='Segoe UI, sans-serif')
    )

    fig.update_yaxes(tickprefix='$', tickformat=',.0f', row=1, col=1)
    fig.update_yaxes(tickformat=',.0f', row=1, col=2)

    return fig


def create_heatmap_correlation(df: pd.DataFrame) -> go.Figure:
    """
    Create a correlation heatmap of numerical variables.

    Args:
        df: DataFrame with numerical columns

    Returns:
        Plotly Figure object
    """
    # Select numerical columns
    numerical_cols = ['listing_price', 'sq_ft', 'bedrooms', 'price_per_sqft',
                      'median_income', 'school_rating', 'crime_index_numeric', 'investment_score']
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    corr_matrix = df[numerical_cols].corr()

    # Create readable labels
    labels = {
        'listing_price': 'Price',
        'sq_ft': 'Sq.Ft',
        'bedrooms': 'Beds',
        'price_per_sqft': 'Price/SqFt',
        'median_income': 'Income',
        'school_rating': 'School',
        'crime_index_numeric': 'Crime',
        'investment_score': 'Inv.Score'
    }

    display_labels = [labels.get(col, col) for col in numerical_cols]

    fig = px.imshow(
        corr_matrix,
        x=display_labels,
        y=display_labels,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='📈 Correlation Matrix',
        zmin=-1, zmax=1
    )

    fig.update_traces(
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={'size': 12}
    )

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_map_visualization(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter map of property locations.

    Args:
        df: DataFrame with latitude, longitude, and property data

    Returns:
        Plotly Figure object
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return go.Figure().update_layout(
            title='Map data not available',
            annotations=[dict(
                text='Coordinate data required for map visualization',
                showarrow=False,
                font=dict(size=14)
            )]
        )

    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color='price_per_sqft',
        size='sq_ft',
        size_max=20,
        hover_name='raw_address',
        hover_data={
            'listing_price': ':$,.0f',
            'sq_ft': ':,.0f',
            'bedrooms': True,
            'school_rating': ':.1f',
            'matched_zip': True,
            'latitude': False,
            'longitude': False
        },
        color_continuous_scale='Viridis',
        title='🗺️ Property Locations',
        zoom=8
    )

    fig.update_layout(
        mapbox_style='open-street-map',
        margin=dict(r=0, t=50, l=0, b=0),
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_price_trend_by_zip(df: pd.DataFrame) -> go.Figure:
    """
    Create a box plot showing price distribution by ZIP code.

    Args:
        df: DataFrame with matched_zip and listing_price columns

    Returns:
        Plotly Figure object
    """
    # Sort by median price
    zip_order = df.groupby('matched_zip')['listing_price'].median().sort_values().index.tolist()

    fig = px.box(
        df,
        x='matched_zip',
        y='listing_price',
        color='crime_index',
        color_discrete_map=CRIME_COLORS,
        title='📦 Price Distribution by ZIP Code',
        labels={'matched_zip': 'ZIP Code', 'listing_price': 'Listing Price ($)'},
        category_orders={'matched_zip': zip_order}
    )

    fig.update_layout(
        template='plotly_white',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.0f',
        xaxis_tickangle=-45,
        font=dict(family='Segoe UI, sans-serif')
    )

    return fig


def create_top_properties_table(
    df: pd.DataFrame,
    top_n: int = 10,
    sort_by: str = 'investment_score'
) -> pd.DataFrame:
    """
    Create a formatted table of top properties.

    Args:
        df: DataFrame with property data
        top_n: Number of top properties to show
        sort_by: Column to sort by

    Returns:
        Formatted DataFrame for display
    """
    if sort_by not in df.columns:
        sort_by = 'listing_price'

    top_df = df.nlargest(top_n, sort_by)[[
        'raw_address', 'matched_zip', 'listing_price', 'sq_ft',
        'bedrooms', 'price_per_sqft', 'school_rating', 'crime_index',
        'investment_score'
    ]].copy()

    # Format columns
    top_df.columns = [
        'Address', 'ZIP Code', 'Price', 'Sq.Ft',
        'Beds', 'Price/SqFt', 'School Rating', 'Crime Level',
        'Investment Score'
    ]

    return top_df


def create_kpi_card_data(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format statistics for KPI card display.

    Args:
        stats: Dictionary of summary statistics

    Returns:
        List of KPI card configurations
    """
    kpi_cards = [
        {
            'title': 'Total Listings',
            'value': f"{stats.get('total_listings', 0):,}",
            'icon': '🏠',
            'delta': None,
            'color': COLORS['primary']
        },
        {
            'title': 'Match Rate',
            'value': f"{stats.get('match_rate', 0):.1f}%",
            'icon': '🎯',
            'delta': None,
            'color': COLORS['success'] if stats.get('match_rate', 0) > 90 else COLORS['warning']
        },
        {
            'title': 'Avg Price',
            'value': f"${stats.get('avg_listing_price', 0):,.0f}",
            'icon': '💰',
            'delta': None,
            'color': COLORS['info']
        },
        {
            'title': 'Avg Price/SqFt',
            'value': f"${stats.get('avg_price_per_sqft', 0):,.0f}",
            'icon': '📐',
            'delta': None,
            'color': COLORS['secondary']
        },
        {
            'title': 'Avg School Rating',
            'value': f"{stats.get('avg_school_rating', 0):.1f}/10",
            'icon': '📚',
            'delta': None,
            'color': COLORS['success']
        },
        {
            'title': 'Avg Investment Score',
            'value': f"{stats.get('avg_investment_score', 0):.1f}",
            'icon': '📈',
            'delta': None,
            'color': COLORS['primary']
        }
    ]

    return kpi_cards


def create_match_quality_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing match quality distribution.

    Args:
        df: DataFrame with match_type and match_score columns

    Returns:
        Plotly Figure object
    """
    match_counts = df['match_type'].value_counts().reset_index()
    match_counts.columns = ['Match Type', 'Count']

    color_map = {
        'exact': COLORS['success'],
        'fuzzy': COLORS['warning'],
        'partial': COLORS['info'],
        'no_match': COLORS['danger']
    }

    fig = px.bar(
        match_counts,
        x='Match Type',
        y='Count',
        color='Match Type',
        color_discrete_map=color_map,
        title='🔗 Data Matching Quality',
        text='Count'
    )

    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        font=dict(family='Segoe UI, sans-serif')
    )

    fig.update_traces(textposition='outside')

    return fig
