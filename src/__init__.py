"""
Source Package Initialization
Author: Shashi Raj
"""

from .data_processing import DataProcessor, create_sample_coordinates
from .visualizations import (
    create_price_distribution_chart,
    create_price_vs_sqft_scatter,
    create_price_per_sqft_by_zip,
    create_school_rating_vs_price,
    create_investment_score_gauge,
    create_crime_index_pie,
    create_income_distribution,
    create_bedrooms_analysis,
    create_heatmap_correlation,
    create_map_visualization,
    create_price_trend_by_zip,
    create_top_properties_table,
    create_kpi_card_data,
    create_match_quality_chart,
    COLORS,
    CRIME_COLORS
)
from .utils import (
    format_currency,
    format_percentage,
    format_number,
    get_rating_emoji,
    get_crime_badge,
    validate_dataframe,
    calculate_investment_metrics,
    get_filter_options,
    create_summary_text,
    get_app_info,
    dataframe_to_download
)

__all__ = [
    'DataProcessor',
    'create_sample_coordinates',
    'create_price_distribution_chart',
    'create_price_vs_sqft_scatter',
    'create_price_per_sqft_by_zip',
    'create_school_rating_vs_price',
    'create_investment_score_gauge',
    'create_crime_index_pie',
    'create_income_distribution',
    'create_bedrooms_analysis',
    'create_heatmap_correlation',
    'create_map_visualization',
    'create_price_trend_by_zip',
    'create_top_properties_table',
    'create_kpi_card_data',
    'create_match_quality_chart',
    'COLORS',
    'CRIME_COLORS',
    'format_currency',
    'format_percentage',
    'format_number',
    'get_rating_emoji',
    'get_crime_badge',
    'validate_dataframe',
    'calculate_investment_metrics',
    'get_filter_options',
    'create_summary_text',
    'get_app_info',
    'dataframe_to_download'
]
