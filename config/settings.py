"""
Configuration Settings for Property Investment Insights Dashboard
Author: Shashi Raj

This module contains all configuration constants and settings.
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Assets directory
ASSETS_DIR = BASE_DIR / "assets"

# Data file paths
DEMOGRAPHICS_FILE = DATA_DIR / "demographics.csv"
LISTINGS_FILE = DATA_DIR / "listings.csv"

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_CONFIG = {
    "title": "Property Investment Insights Dashboard",
    "subtitle": "Single Source of Truth for Property Analysis",
    "page_icon": "🏠",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "author": "Shashi Raj",
    "version": "1.0.0"
}

# ============================================================================
# DATA PROCESSING SETTINGS
# ============================================================================

DATA_CONFIG = {
    # Fuzzy matching threshold (0-100)
    "fuzzy_threshold": 80,

    # Minimum match score to consider valid
    "min_match_score": 60,

    # Default investment score weights
    "investment_weights": {
        "price_per_sqft": 0.25,
        "school_rating": 0.30,
        "crime_index": 0.25,
        "price_to_income": 0.20
    },

    # Crime index mapping
    "crime_index_map": {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    # Color scheme
    "colors": {
        "primary": "#667eea",
        "secondary": "#764ba2",
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "info": "#17a2b8",
        "light": "#f8f9fa",
        "dark": "#343a40",
        "background": "#0e1117",
        "card_bg": "#1e1e1e"
    },

    # Chart theme
    "chart_theme": "plotly_white",

    # Font family
    "font_family": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif",

    # Default chart height
    "default_chart_height": 400,

    # Map configuration
    "map_config": {
        "default_zoom": 8,
        "style": "open-street-map"
    }
}

# ============================================================================
# FILTER DEFAULTS
# ============================================================================

FILTER_DEFAULTS = {
    "price_step": 10000,
    "sqft_step": 100,
    "school_rating_step": 0.5,
    "investment_score_step": 5
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    # Time to live for cached data (in seconds)
    "ttl": 3600,  # 1 hour

    # Maximum entries in cache
    "max_entries": 100
}

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

EXPORT_CONFIG = {
    "formats": ["csv", "json"],
    "default_format": "csv",
    "filename_prefix": "property_insights"
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "app.log"
}
