"""
Utility Module for Property Investment Insights Dashboard
Author: Shashi Raj

This module provides:
- Caching utilities
- Data formatters
- Helper functions
- Export utilities
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import json


def format_currency(value: Union[int, float], decimals: int = 0) -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return "N/A"

    if decimals == 0:
        return f"${value:,.0f}"
    return f"${value:,.{decimals}f}"


def format_percentage(value: Union[int, float], decimals: int = 1) -> str:
    """
    Format a number as percentage.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """
    Format a number with thousands separator.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"

    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def calculate_percentile(series: pd.Series, value: float) -> float:
    """
    Calculate the percentile rank of a value in a series.

    Args:
        series: Pandas Series of values
        value: Value to find percentile for

    Returns:
        Percentile rank (0-100)
    """
    return (series < value).sum() / len(series) * 100


def get_rating_emoji(rating: float, max_rating: float = 10) -> str:
    """
    Get an emoji representation of a rating.

    Args:
        rating: The rating value
        max_rating: Maximum possible rating

    Returns:
        Emoji string
    """
    if pd.isna(rating):
        return "❓"

    ratio = rating / max_rating
    if ratio >= 0.8:
        return "⭐⭐⭐⭐⭐"
    elif ratio >= 0.6:
        return "⭐⭐⭐⭐"
    elif ratio >= 0.4:
        return "⭐⭐⭐"
    elif ratio >= 0.2:
        return "⭐⭐"
    else:
        return "⭐"


def get_crime_badge(crime_level: str) -> str:
    """
    Get a badge for crime level.

    Args:
        crime_level: Crime level string (Low, Medium, High)

    Returns:
        HTML badge string
    """
    colors = {
        'Low': '#28a745',
        'Medium': '#ffc107',
        'High': '#dc3545'
    }
    color = colors.get(crime_level, '#6c757d')
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{crime_level}</span>'


def generate_data_hash(df: pd.DataFrame) -> str:
    """
    Generate a hash for a DataFrame to detect changes.

    Args:
        df: DataFrame to hash

    Returns:
        MD5 hash string
    """
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


def dataframe_to_download(df: pd.DataFrame, format: str = 'csv') -> bytes:
    """
    Convert DataFrame to downloadable format.

    Args:
        df: DataFrame to convert
        format: Output format ('csv' or 'json')

    Returns:
        Bytes of the converted data
    """
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8')
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str]
) -> Dict[str, Any]:
    """
    Validate a DataFrame has required columns and data quality.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'missing_columns': [],
        'null_counts': {},
        'row_count': len(df),
        'warnings': []
    }

    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            result['missing_columns'].append(col)
            result['valid'] = False

    # Check null values
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            result['null_counts'][col] = null_count
            if null_count / len(df) > 0.5:
                result['warnings'].append(f"Column '{col}' has >50% null values")

    return result


def calculate_investment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key investment metrics from the merged dataset.

    Args:
        df: Merged DataFrame with property and demographic data

    Returns:
        Dictionary of investment metrics
    """
    metrics = {}

    if df.empty:
        return metrics

    # Price metrics
    metrics['price_stats'] = {
        'mean': df['listing_price'].mean(),
        'median': df['listing_price'].median(),
        'std': df['listing_price'].std(),
        'min': df['listing_price'].min(),
        'max': df['listing_price'].max()
    }

    # Square footage metrics
    metrics['sqft_stats'] = {
        'mean': df['sq_ft'].mean(),
        'median': df['sq_ft'].median(),
        'min': df['sq_ft'].min(),
        'max': df['sq_ft'].max()
    }

    # Price per sqft metrics
    if 'price_per_sqft' in df.columns:
        metrics['price_per_sqft_stats'] = {
            'mean': df['price_per_sqft'].mean(),
            'median': df['price_per_sqft'].median(),
            'min': df['price_per_sqft'].min(),
            'max': df['price_per_sqft'].max()
        }

    # Investment score distribution
    if 'investment_score' in df.columns:
        metrics['investment_score_distribution'] = {
            'excellent': len(df[df['investment_score'] >= 80]),
            'good': len(df[(df['investment_score'] >= 60) & (df['investment_score'] < 80)]),
            'fair': len(df[(df['investment_score'] >= 40) & (df['investment_score'] < 60)]),
            'poor': len(df[df['investment_score'] < 40])
        }

    # Top ZIP codes by investment score
    if 'matched_zip' in df.columns and 'investment_score' in df.columns:
        top_zips = df.groupby('matched_zip')['investment_score'].mean().nlargest(5)
        metrics['top_zip_codes'] = top_zips.to_dict()

    return metrics


def get_filter_options(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract filter options from the dataset.

    Args:
        df: DataFrame to extract options from

    Returns:
        Dictionary of filter options
    """
    options = {}

    if 'matched_zip' in df.columns:
        options['zip_codes'] = sorted(df['matched_zip'].dropna().unique().tolist())

    if 'listing_price' in df.columns:
        options['price_range'] = (
            float(df['listing_price'].min()),
            float(df['listing_price'].max())
        )

    if 'sq_ft' in df.columns:
        options['sqft_range'] = (
            float(df['sq_ft'].min()),
            float(df['sq_ft'].max())
        )

    if 'bedrooms' in df.columns:
        options['bedrooms'] = sorted(df['bedrooms'].dropna().unique().tolist())

    if 'crime_index' in df.columns:
        options['crime_levels'] = df['crime_index'].dropna().unique().tolist()

    if 'school_rating' in df.columns:
        options['school_rating_range'] = (
            float(df['school_rating'].min()),
            float(df['school_rating'].max())
        )

    return options


def create_summary_text(stats: Dict[str, Any]) -> str:
    """
    Create a summary text from statistics.

    Args:
        stats: Dictionary of summary statistics

    Returns:
        Formatted summary text
    """
    lines = [
        "📊 **Property Investment Summary**",
        "",
        f"• **Total Listings:** {stats.get('total_listings', 'N/A'):,}",
        f"• **Matched Records:** {stats.get('matched_listings', 'N/A'):,} ({stats.get('match_rate', 0):.1f}%)",
        f"• **Average Price:** {format_currency(stats.get('avg_listing_price', 0))}",
        f"• **Median Price:** {format_currency(stats.get('median_listing_price', 0))}",
        f"• **Avg Price/Sq.Ft:** {format_currency(stats.get('avg_price_per_sqft', 0))}",
        f"• **Avg School Rating:** {stats.get('avg_school_rating', 0):.1f}/10",
        f"• **Unique ZIP Codes:** {stats.get('unique_zip_codes', 'N/A')}",
    ]

    return "\n".join(lines)


def get_app_info() -> Dict[str, str]:
    """
    Get application information.

    Returns:
        Dictionary with app information
    """
    return {
        'name': 'Property Investment Insights Dashboard',
        'version': '1.0.0',
        'author': 'Shashi Raj',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'description': 'A Streamlit-based dashboard for property investment analysis with fuzzy matching capabilities.'
    }
