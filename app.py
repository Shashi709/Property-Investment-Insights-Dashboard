"""
Property Investment Insights Dashboard - Main Application
Author: Shashi Raj

This is the main Streamlit application that provides an interactive dashboard
for property investment analysis with fuzzy matching capabilities.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom modules
from src.data_processing import DataProcessor, create_sample_coordinates
from src.visualizations import (
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
    COLORS
)
from src.utils import (
    format_currency,
    format_percentage,
    format_number,
    get_filter_options,
    create_summary_text,
    get_app_info,
    dataframe_to_download
)
from config.settings import (
    APP_CONFIG,
    DATA_CONFIG,
    UI_CONFIG,
    FILTER_DEFAULTS,
    DEMOGRAPHICS_FILE,
    LISTINGS_FILE
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    """Load custom CSS styles for the dashboard."""
    css_file = PROJECT_ROOT / "assets" / "styles.css"

    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Inline CSS if file doesn't exist
        st.markdown("""
        <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Header styling */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }

        .dashboard-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }

        .dashboard-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }

        /* KPI Card styling */
        .kpi-card {
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #3d3d3d;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }

        .kpi-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #667eea;
            margin: 0.5rem 0;
        }

        .kpi-title {
            font-size: 0.9rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Section headers */
        .section-header {
            border-left: 4px solid #667eea;
            padding-left: 1rem;
            margin: 2rem 0 1rem 0;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }

        /* Filter section */
        .filter-section {
            background: #1e1e1e;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* Data table styling */
        .dataframe {
            font-size: 0.85rem !important;
        }

        /* Footer */
        .dashboard-footer {
            text-align: center;
            padding: 2rem;
            color: #888;
            border-top: 1px solid #3d3d3d;
            margin-top: 3rem;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
            font-weight: 600;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
        }

        /* Chart containers */
        .chart-container {
            background: #1e1e1e;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            border: 1px solid #667eea40;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* Success badge */
        .success-badge {
            background-color: #28a745;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* Warning badge */
        .warning-badge {
            background-color: #ffc107;
            color: #333;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        /* Enhanced Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            padding: 0.5rem;
            gap: 0.5rem;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.8rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar enhancements */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        }
        
        section[data-testid="stSidebar"] h3 {
            color: #667eea !important;
            font-size: 0.9rem;
            letter-spacing: 1px;
        }
        
        /* Plotly chart backgrounds */
        .js-plotly-plot {
            border-radius: 15px;
            overflow: hidden;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 15px;
            overflow: hidden;
        }
        
        /* Download buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
            border: none !important;
            border-radius: 10px;
            font-weight: 600;
        }
        
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data(ttl=3600)
def load_and_process_data():
    """
    Load and process data with caching for performance.

    Returns:
        Tuple of (merged_df, stats, processor)
    """
    processor = DataProcessor(fuzzy_threshold=DATA_CONFIG["fuzzy_threshold"])

    try:
        # Load data files
        demographics_df = processor.load_demographics(str(DEMOGRAPHICS_FILE))
        listings_df = processor.load_listings(str(LISTINGS_FILE))

        # Merge data with fuzzy matching
        merged_df = processor.merge_data()

        # Add coordinates for mapping
        merged_df = create_sample_coordinates(merged_df)

        # Calculate statistics
        stats = processor.get_summary_statistics()

        return merged_df, stats, True

    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        return pd.DataFrame(), {}, False
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame(), {}, False


# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

def render_sidebar(df: pd.DataFrame):
    """
    Render the sidebar with filters and controls.

    Args:
        df: DataFrame with filter options

    Returns:
        Dictionary of filter values
    """
    with st.sidebar:
        # Theme Toggle
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2>🏠 Property Insights</h2>
            <p style="color: #888; font-size: 0.9rem;">Investment Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dark/Light Mode Toggle
        theme_mode = st.toggle("🌙 Dark Mode", value=True, key="theme_toggle")
        
        if not theme_mode:
            st.markdown("""
            <style>
            /* Light Mode Styles */
            .stApp {
                background-color: #ffffff !important;
                color: #1a1a1a !important;
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
            }
            section[data-testid="stSidebar"] * {
                color: #1a1a1a !important;
            }
            .kpi-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
                border: 1px solid #dee2e6 !important;
                color: #1a1a1a !important;
            }
            .kpi-title {
                color: #495057 !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, div {
                color: #1a1a1a !important;
            }
            .stMarkdown {
                color: #1a1a1a !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

        # Get filter options
        filter_options = get_filter_options(df)

        filters = {}

        # ---- ZIP Code Filter ----
        st.markdown("### 📍 Location Filter")
        zip_codes = filter_options.get('zip_codes', [])
        if zip_codes:
            filters['zip_codes'] = st.multiselect(
                "Select ZIP Codes",
                options=zip_codes,
                default=[],
                help="Filter properties by ZIP code"
            )

        st.divider()

        # ---- Price Range Filter ----
        st.markdown("### 💰 Price Filter")
        price_range = filter_options.get('price_range', (0, 1000000))
        filters['price_range'] = st.slider(
            "Listing Price Range",
            min_value=int(price_range[0]),
            max_value=int(price_range[1]),
            value=(int(price_range[0]), int(price_range[1])),
            step=FILTER_DEFAULTS["price_step"],
            format="$%d"
        )

        st.divider()

        # ---- Square Footage Filter ----
        st.markdown("### 📐 Size Filter")
        sqft_range = filter_options.get('sqft_range', (500, 5000))
        filters['sqft_range'] = st.slider(
            "Square Footage Range",
            min_value=int(sqft_range[0]),
            max_value=int(sqft_range[1]),
            value=(int(sqft_range[0]), int(sqft_range[1])),
            step=FILTER_DEFAULTS["sqft_step"],
            format="%d sq.ft"
        )

        st.divider()

        # ---- Bedroom Filter ----
        st.markdown("### 🛏️ Bedrooms")
        bedrooms = filter_options.get('bedrooms', [1, 2, 3, 4, 5])
        filters['bedrooms'] = st.multiselect(
            "Number of Bedrooms",
            options=sorted([int(b) for b in bedrooms if pd.notna(b)]),
            default=[],
            help="Filter by bedroom count"
        )

        st.divider()

        # ---- Demographics Filters ----
        st.markdown("### 📊 Demographics")

        # School Rating
        school_range = filter_options.get('school_rating_range', (0, 10))
        filters['school_rating_min'] = st.slider(
            "Minimum School Rating",
            min_value=float(school_range[0]),
            max_value=float(school_range[1]),
            value=float(school_range[0]),
            step=FILTER_DEFAULTS["school_rating_step"]
        )

        # Crime Level
        crime_levels = filter_options.get('crime_levels', ['Low', 'Medium', 'High'])
        filters['crime_index'] = st.multiselect(
            "Crime Level",
            options=crime_levels,
            default=crime_levels,
            help="Filter by neighborhood crime level"
        )

        st.divider()

        # ---- Investment Score Filter ----
        st.markdown("### 📈 Investment Score")
        filters['investment_score_min'] = st.slider(
            "Minimum Investment Score",
            min_value=0,
            max_value=100,
            value=0,
            step=FILTER_DEFAULTS["investment_score_step"]
        )

        st.divider()

        # ---- About Section ----
        with st.expander("ℹ️ About This Dashboard"):
            app_info = get_app_info()
            st.markdown(f"""
            **{app_info['name']}**

            Version: {app_info['version']}

            **Author:** {app_info['author']}

            ---

            This dashboard provides comprehensive property investment analysis
            with advanced fuzzy matching to merge messy listing data with
            structured demographic information.
            """)

        return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply filters to the DataFrame.

    Args:
        df: Original DataFrame
        filters: Dictionary of filter values

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # ZIP Code filter
    if filters.get('zip_codes'):
        filtered_df = filtered_df[filtered_df['matched_zip'].isin(filters['zip_codes'])]

    # Price range filter
    if filters.get('price_range'):
        filtered_df = filtered_df[
            (filtered_df['listing_price'] >= filters['price_range'][0]) &
            (filtered_df['listing_price'] <= filters['price_range'][1])
        ]

    # Square footage filter
    if filters.get('sqft_range'):
        filtered_df = filtered_df[
            (filtered_df['sq_ft'] >= filters['sqft_range'][0]) &
            (filtered_df['sq_ft'] <= filters['sqft_range'][1])
        ]

    # Bedroom filter
    if filters.get('bedrooms'):
        filtered_df = filtered_df[filtered_df['bedrooms'].isin(filters['bedrooms'])]

    # School rating filter
    if filters.get('school_rating_min') is not None:
        filtered_df = filtered_df[filtered_df['school_rating'] >= filters['school_rating_min']]

    # Crime level filter
    if filters.get('crime_index'):
        filtered_df = filtered_df[filtered_df['crime_index'].isin(filters['crime_index'])]

    # Investment score filter
    if filters.get('investment_score_min') is not None:
        filtered_df = filtered_df[filtered_df['investment_score'] >= filters['investment_score_min']]

    return filtered_df


# ============================================================================
# MAIN DASHBOARD COMPONENTS
# ============================================================================

def render_header():
    """Render the dashboard header."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%);
            animation: pulse 4s ease-in-out infinite;
        "></div>
        <h1 style="
            margin: 0;
            font-size: 2.8rem;
            font-weight: 800;
            color: white;
            text-shadow: 2px 4px 8px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        ">🏠 Property Investment Insights Dashboard</h1>
        <p style="
            margin: 1rem 0 0 0;
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 400;
            position: relative;
            z-index: 1;
        ">Single Source of Truth for Property Value vs. Neighborhood Demographics</p>
        <p style="
            margin: 0.5rem 0 0 0;
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.8);
            position: relative;
            z-index: 1;
        ">Author: <strong>Shashi Raj</strong></p>
    </div>
    <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
    </style>
    """, unsafe_allow_html=True)


def render_kpi_cards(stats: dict, filtered_count: int):
    """
    Render KPI metric cards.

    Args:
        stats: Dictionary of statistics
        filtered_count: Number of filtered records
    """
    # Create 2 rows of 3 columns each for 6 KPIs
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    # KPI data
    kpis = [
        ('🏠', stats.get('total_listings', 0), 'Total Listings', '#667eea', '#764ba2', 'number'),
        ('🎯', stats.get('match_rate', 0), 'Match Rate', '#f093fb', '#f5576c', 'percent'),
        ('💰', stats.get('avg_listing_price', 0), 'Avg Price', '#4facfe', '#00f2fe', 'currency'),
        ('📐', stats.get('avg_price_per_sqft', 0), 'Avg Price/SqFt', '#43e97b', '#38f9d7', 'currency'),
        ('📚', stats.get('avg_school_rating', 0), 'Avg School Rating', '#fa709a', '#fee140', 'rating'),
        ('📈', stats.get('avg_investment_score', 0), 'Avg Investment Score', '#a8edea', '#fed6e3', 'score')
    ]
    
    columns = [col1, col2, col3, col4, col5, col6]
    
    for idx, (icon, value, title, color1, color2, value_type) in enumerate(kpis):
        with columns[idx]:
            # Format value based on type
            if value_type == 'number':
                formatted_value = f"{int(value):,}"
            elif value_type == 'percent':
                formatted_value = f"{value:.1f}%"
            elif value_type == 'currency':
                formatted_value = f"${int(value):,}"
            elif value_type == 'rating':
                formatted_value = f"{value:.1f}/10"
            else:  # score
                formatted_value = f"{value:.1f}"
            
            st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%); 
                            border-radius: 16px; padding: 1.5rem 1rem; text-align: center; 
                            border: 1px solid rgba(102, 126, 234, 0.3); 
                            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); margin-bottom: 1rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; 
                                background: linear-gradient(135deg, {color1} 0%, {color2} 100%); 
                                -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                                background-clip: text; margin: 0.5rem 0;">{formatted_value}</div>
                    <div style="font-size: 0.75rem; color: #888; text-transform: uppercase; 
                                letter-spacing: 1.5px; font-weight: 600;">{title}</div>
                </div>
            """, unsafe_allow_html=True)

    # Show filtered count
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0 2rem 0;
        font-size: 1rem;
    ">
        <strong style="color: #667eea;">📋 Filtered Results:</strong> 
        <span style="color: #fff;">{filtered_count:,} properties match your criteria</span>
    </div>
    """, unsafe_allow_html=True)


def render_overview_tab(df: pd.DataFrame, stats: dict):
    """
    Render the Overview tab content.

    Args:
        df: Filtered DataFrame
        stats: Statistics dictionary
    """
    # Row 1: Price Distribution and Price vs SqFt
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_price_distribution_chart(df, color_by='crime_index'),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_price_vs_sqft_scatter(df, color_by='school_rating', trendline=True),
            use_container_width=True
        )

    # Row 2: Price per SqFt by ZIP and School Rating vs Price
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_price_per_sqft_by_zip(df),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_school_rating_vs_price(df),
            use_container_width=True
        )


def render_demographics_tab(df: pd.DataFrame):
    """
    Render the Demographics tab content.

    Args:
        df: Filtered DataFrame
    """
    # Row 1: Crime Index and Income Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_crime_index_pie(df),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_income_distribution(df),
            use_container_width=True
        )

    # Row 2: Bedroom Analysis
    st.plotly_chart(
        create_bedrooms_analysis(df),
        use_container_width=True
    )

    # Row 3: Price Distribution by ZIP
    st.plotly_chart(
        create_price_trend_by_zip(df),
        use_container_width=True
    )


def render_analytics_tab(df: pd.DataFrame, stats: dict):
    """
    Render the Analytics tab content.

    Args:
        df: Filtered DataFrame
        stats: Statistics dictionary
    """
    # Row 1: Investment Score Gauge and Correlation Matrix
    col1, col2 = st.columns([1, 2])

    with col1:
        avg_score = df['investment_score'].mean() if len(df) > 0 else 0
        st.plotly_chart(
            create_investment_score_gauge(avg_score),
            use_container_width=True
        )

        # Investment Score Legend
        st.markdown("""
        **Investment Score Components:**
        - 🏷️ Price per Sq.Ft (25%)
        - 📚 School Rating (30%)
        - 🔒 Crime Index (25%)
        - 💵 Price-to-Income Ratio (20%)
        """)

    with col2:
        st.plotly_chart(
            create_heatmap_correlation(df),
            use_container_width=True
        )

    # Row 2: Match Quality Chart
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_match_quality_chart(df),
            use_container_width=True
        )

    with col2:
        # Top Properties Table
        st.markdown("### 🏆 Top Investment Properties")
        top_properties = create_top_properties_table(df, top_n=10)
        st.dataframe(
            top_properties,
            use_container_width=True,
            hide_index=True
        )


def render_map_tab(df: pd.DataFrame):
    """
    Render the Map tab content.

    Args:
        df: Filtered DataFrame
    """
    st.markdown("### 🗺️ Property Location Map")
    st.info("📍 Properties are clustered by ZIP code. Colors represent price per square foot.")

    st.plotly_chart(
        create_map_visualization(df),
        use_container_width=True,
        height=600
    )

    # ZIP Code Summary Table
    st.markdown("### 📊 ZIP Code Summary")

    zip_summary = df.groupby('matched_zip').agg({
        'listing_price': ['mean', 'count'],
        'sq_ft': 'mean',
        'price_per_sqft': 'mean',
        'school_rating': 'first',
        'crime_index': 'first',
        'investment_score': 'mean'
    }).round(2)

    zip_summary.columns = ['Avg Price', 'Count', 'Avg SqFt', 'Avg $/SqFt', 'School Rating', 'Crime Level', 'Avg Inv. Score']
    zip_summary = zip_summary.sort_values('Avg Inv. Score', ascending=False)

    st.dataframe(zip_summary, use_container_width=True)


def render_data_tab(df: pd.DataFrame):
    """
    Render the Data Explorer tab content.

    Args:
        df: Filtered DataFrame
    """
    st.markdown("### 📋 Property Data Explorer")

    # Column selector
    all_columns = df.columns.tolist()
    display_columns = st.multiselect(
        "Select columns to display",
        options=all_columns,
        default=['raw_address', 'matched_zip', 'listing_price', 'sq_ft',
                 'bedrooms', 'price_per_sqft', 'school_rating', 'crime_index',
                 'investment_score', 'match_type']
    )

    # Sort options
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox("Sort by", options=display_columns, index=display_columns.index('listing_price') if 'listing_price' in display_columns else 0)
    with col2:
        sort_order = st.selectbox("Sort order", options=['Descending', 'Ascending'])

    # Apply sorting
    ascending = sort_order == 'Ascending'
    sorted_df = df[display_columns].sort_values(by=sort_by, ascending=ascending)

    # Display data
    st.dataframe(
        sorted_df,
        use_container_width=True,
        height=500
    )

    # Export options
    st.markdown("### 📥 Export Data")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        csv_data = dataframe_to_download(sorted_df, 'csv')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="property_insights_export.csv",
            mime="text/csv"
        )

    with col2:
        json_data = dataframe_to_download(sorted_df, 'json')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="property_insights_export.json",
            mime="application/json"
        )


def render_footer():
    """Render the dashboard footer."""
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2.5rem;
        margin-top: 3rem;
        background: linear-gradient(180deg, transparent 0%, rgba(102, 126, 234, 0.1) 100%);
        border-top: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px 20px 0 0;
    ">
        <p style="margin: 0; color: #888; line-height: 2;">
            <span style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                font-size: 1.1rem;
            ">Property Investment Insights Dashboard</span> | Version 1.0.0<br>
            Developed by <strong style="color: #667eea;">Shashi Raj</strong> | 
            <span style="font-size: 0.9rem;">© 2025 | All Rights Reserved</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""

    # Load custom CSS
    load_custom_css()

    # Render header
    render_header()

    # Load data
    with st.spinner("🔄 Loading and processing data with fuzzy matching..."):
        df, stats, success = load_and_process_data()

    if not success or df.empty:
        st.error("""
        ### ❌ Unable to Load Data

        Please ensure the following files exist in the `data/` directory:
        - `demographics.csv`
        - `listings.csv`

        Check the file paths and try again.
        """)
        return

    # Render sidebar and get filters
    filters = render_sidebar(df)

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Update stats for filtered data
    if len(filtered_df) > 0:
        processor = DataProcessor()
        filtered_stats = processor.get_summary_statistics(filtered_df)
    else:
        filtered_stats = stats

    # Render KPI cards
    render_kpi_cards(filtered_stats if len(filtered_df) > 0 else stats, len(filtered_df))

    # Check if any data after filtering
    if len(filtered_df) == 0:
        st.warning("⚠️ No properties match your filter criteria. Please adjust the filters.")
        return

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "👥 Demographics",
        "📈 Analytics",
        "🗺️ Map View",
        "📋 Data Explorer"
    ])

    with tab1:
        render_overview_tab(filtered_df, filtered_stats)

    with tab2:
        render_demographics_tab(filtered_df)

    with tab3:
        render_analytics_tab(filtered_df, filtered_stats)

    with tab4:
        render_map_tab(filtered_df)

    with tab5:
        render_data_tab(filtered_df)

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
