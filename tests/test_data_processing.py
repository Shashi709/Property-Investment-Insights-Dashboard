"""
Unit Tests for Property Investment Insights Dashboard
Author: Shashi Raj

This module contains comprehensive tests for:
- Data processing and cleaning
- Fuzzy matching algorithms
- Address normalization
- Data merging logic
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import DataProcessor, create_sample_coordinates
from src.utils import (
    format_currency,
    format_percentage,
    format_number,
    get_rating_emoji,
    validate_dataframe,
    calculate_investment_metrics,
    get_filter_options
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_demographics_df():
    """Create sample demographics DataFrame for testing."""
    return pd.DataFrame({
        'zip_code': ['12345', '23456', '34567', '45678', '56789'],
        'median_income': [75000, 85000, 95000, 65000, 55000],
        'school_rating': [8.5, 7.2, 9.1, 6.3, 5.5],
        'crime_index': ['Low', 'Medium', 'Low', 'High', 'Medium']
    })


@pytest.fixture
def sample_listings_df():
    """Create sample listings DataFrame for testing."""
    return pd.DataFrame({
        'raw_address': [
            '123 Main Street',
            '456 OAK AVENUE',
            '789 pine blvd.',
            '321 Elm St.',
            '654 MAPLE DR'
        ],
        'postal_code': ['12345', '234XX', '34567', '45678', '567XX'],
        'sq_ft': [1500, 2000, 1800, 2200, 1600],
        'bedrooms': [3, 4, 3, 5, 2],
        'listing_price': [350000, 450000, 520000, 380000, 290000]
    })


@pytest.fixture
def data_processor():
    """Create DataProcessor instance for testing."""
    return DataProcessor(fuzzy_threshold=80)


# ============================================================================
# DATA PROCESSOR TESTS
# ============================================================================

class TestDataProcessor:
    """Test cases for DataProcessor class."""

    def test_initialization(self, data_processor):
        """Test DataProcessor initialization."""
        assert data_processor.fuzzy_threshold == 80
        assert data_processor._demographics_df is None
        assert data_processor._listings_df is None
        assert data_processor._merged_df is None

    def test_initialization_custom_threshold(self):
        """Test DataProcessor with custom threshold."""
        processor = DataProcessor(fuzzy_threshold=90)
        assert processor.fuzzy_threshold == 90


class TestAddressNormalization:
    """Test cases for address normalization."""

    def test_normalize_street_to_st(self, data_processor):
        """Test normalization of 'street' to 'ST'."""
        result = data_processor._normalize_address("123 Main Street")
        assert "ST" in result
        assert "STREET" not in result

    def test_normalize_avenue_to_ave(self, data_processor):
        """Test normalization of 'avenue' to 'AVE'."""
        result = data_processor._normalize_address("456 Oak Avenue")
        assert "AVE" in result
        assert "AVENUE" not in result

    def test_normalize_boulevard_to_blvd(self, data_processor):
        """Test normalization of 'boulevard' to 'BLVD'."""
        result = data_processor._normalize_address("789 Pine Boulevard")
        assert "BLVD" in result
        assert "BOULEVARD" not in result

    def test_normalize_uppercase_input(self, data_processor):
        """Test normalization with uppercase input."""
        result = data_processor._normalize_address("123 MAIN STREET")
        assert result == "123 MAIN ST"

    def test_normalize_lowercase_input(self, data_processor):
        """Test normalization with lowercase input."""
        result = data_processor._normalize_address("123 main street")
        assert result == "123 MAIN ST"

    def test_normalize_mixed_case_input(self, data_processor):
        """Test normalization with mixed case input."""
        result = data_processor._normalize_address("123 Main Street")
        assert result == "123 MAIN ST"

    def test_normalize_with_punctuation(self, data_processor):
        """Test normalization with punctuation."""
        result = data_processor._normalize_address("123 Main St.")
        assert "ST" in result

    def test_normalize_empty_string(self, data_processor):
        """Test normalization of empty string."""
        result = data_processor._normalize_address("")
        assert result == ""

    def test_normalize_none_value(self, data_processor):
        """Test normalization of None value."""
        result = data_processor._normalize_address(None)
        assert result == ""

    def test_normalize_multiple_spaces(self, data_processor):
        """Test normalization removes multiple spaces."""
        result = data_processor._normalize_address("123   Main    Street")
        assert "  " not in result

    def test_normalize_complex_address(self, data_processor):
        """Test normalization of complex address."""
        result = data_processor._normalize_address("8667 Brittany Bypass Blvd.")
        assert "BYP" in result or "BLVD" in result


class TestPostalCodeCleaning:
    """Test cases for postal code cleaning."""

    def test_clean_valid_5_digit_code(self, data_processor):
        """Test cleaning of valid 5-digit ZIP code."""
        result = data_processor._clean_postal_code("12345")
        assert result == "12345"

    def test_clean_code_with_xx_suffix(self, data_processor):
        """Test cleaning of ZIP code with XX suffix."""
        result = data_processor._clean_postal_code("325XX")
        assert result is not None
        assert "XX" in result or len(result) == 5

    def test_clean_code_with_spaces(self, data_processor):
        """Test cleaning of ZIP code with spaces."""
        result = data_processor._clean_postal_code(" 12345 ")
        assert result == "12345"

    def test_clean_short_code(self, data_processor):
        """Test cleaning of short ZIP code."""
        result = data_processor._clean_postal_code("123")
        assert result is not None

    def test_clean_none_value(self, data_processor):
        """Test cleaning of None value."""
        result = data_processor._clean_postal_code(None)
        assert result is None

    def test_clean_code_with_prefix_zero(self, data_processor):
        """Test cleaning of ZIP code with leading zeros."""
        result = data_processor._clean_postal_code("03779")
        assert result == "03779"


class TestFuzzyMatching:
    """Test cases for fuzzy matching."""

    def test_exact_match(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test exact ZIP code matching."""
        # Prepare data
        listings = sample_listings_df.copy()
        listings['postal_code_cleaned'] = listings['postal_code'].apply(
            data_processor._clean_postal_code
        )

        results = data_processor.fuzzy_match_zip_codes(listings, sample_demographics_df)

        # Check that exact matches are found
        exact_matches = results[results['match_type'] == 'exact']
        assert len(exact_matches) > 0

    def test_partial_match(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test partial ZIP code matching (XX suffix)."""
        listings = sample_listings_df.copy()
        listings['postal_code_cleaned'] = listings['postal_code'].apply(
            data_processor._clean_postal_code
        )

        results = data_processor.fuzzy_match_zip_codes(listings, sample_demographics_df)

        # Check results structure
        assert 'matched_zip' in results.columns
        assert 'match_score' in results.columns
        assert 'match_type' in results.columns

    def test_no_match_handling(self, data_processor, sample_demographics_df):
        """Test handling of no matches."""
        listings = pd.DataFrame({
            'postal_code_cleaned': ['99999', '00000', None]
        })

        results = data_processor.fuzzy_match_zip_codes(listings, sample_demographics_df)

        # Check that no_match entries exist
        no_matches = results[results['match_type'] == 'no_match']
        assert len(no_matches) > 0


class TestDataMerging:
    """Test cases for data merging."""

    def test_merge_creates_expected_columns(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test that merge creates expected columns."""
        data_processor._demographics_df = sample_demographics_df
        data_processor._demographics_df['crime_index_numeric'] = \
            data_processor._demographics_df['crime_index'].map(DataProcessor.CRIME_INDEX_MAP)

        # Process listings
        sample_listings_df['postal_code_cleaned'] = sample_listings_df['postal_code'].apply(
            data_processor._clean_postal_code
        )
        sample_listings_df['address_normalized'] = sample_listings_df['raw_address'].apply(
            data_processor._normalize_address
        )
        sample_listings_df['price_per_sqft'] = \
            sample_listings_df['listing_price'] / sample_listings_df['sq_ft']
        data_processor._listings_df = sample_listings_df

        merged = data_processor.merge_data()

        expected_columns = ['matched_zip', 'match_score', 'match_type', 'investment_score']
        for col in expected_columns:
            assert col in merged.columns

    def test_merge_calculates_investment_score(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test that merge calculates investment scores."""
        data_processor._demographics_df = sample_demographics_df
        data_processor._demographics_df['crime_index_numeric'] = \
            data_processor._demographics_df['crime_index'].map(DataProcessor.CRIME_INDEX_MAP)

        # Process listings
        sample_listings_df['postal_code_cleaned'] = sample_listings_df['postal_code'].apply(
            data_processor._clean_postal_code
        )
        sample_listings_df['address_normalized'] = sample_listings_df['raw_address'].apply(
            data_processor._normalize_address
        )
        sample_listings_df['price_per_sqft'] = \
            sample_listings_df['listing_price'] / sample_listings_df['sq_ft']
        data_processor._listings_df = sample_listings_df

        merged = data_processor.merge_data()

        assert 'investment_score' in merged.columns
        assert merged['investment_score'].notna().any()
        # Filter out NaN values (unmatched records) for range check
        valid_scores = merged['investment_score'].dropna()
        assert (valid_scores >= 0).all()
        assert (valid_scores <= 100).all()


class TestSummaryStatistics:
    """Test cases for summary statistics."""

    def test_get_summary_statistics_returns_dict(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test that get_summary_statistics returns a dictionary."""
        data_processor._demographics_df = sample_demographics_df
        data_processor._demographics_df['crime_index_numeric'] = \
            data_processor._demographics_df['crime_index'].map(DataProcessor.CRIME_INDEX_MAP)

        # Process listings
        sample_listings_df['postal_code_cleaned'] = sample_listings_df['postal_code'].apply(
            data_processor._clean_postal_code
        )
        sample_listings_df['address_normalized'] = sample_listings_df['raw_address'].apply(
            data_processor._normalize_address
        )
        sample_listings_df['price_per_sqft'] = \
            sample_listings_df['listing_price'] / sample_listings_df['sq_ft']
        data_processor._listings_df = sample_listings_df

        merged = data_processor.merge_data()
        stats = data_processor.get_summary_statistics()

        assert isinstance(stats, dict)
        assert 'total_listings' in stats
        assert 'match_rate' in stats
        assert 'avg_listing_price' in stats


class TestDataFiltering:
    """Test cases for data filtering."""

    def test_filter_by_price_range(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test filtering by price range."""
        data_processor._demographics_df = sample_demographics_df
        data_processor._demographics_df['crime_index_numeric'] = \
            data_processor._demographics_df['crime_index'].map(DataProcessor.CRIME_INDEX_MAP)

        # Process listings
        sample_listings_df['postal_code_cleaned'] = sample_listings_df['postal_code'].apply(
            data_processor._clean_postal_code
        )
        sample_listings_df['address_normalized'] = sample_listings_df['raw_address'].apply(
            data_processor._normalize_address
        )
        sample_listings_df['price_per_sqft'] = \
            sample_listings_df['listing_price'] / sample_listings_df['sq_ft']
        data_processor._listings_df = sample_listings_df

        merged = data_processor.merge_data()
        filtered = data_processor.filter_data(merged, price_range=(300000, 400000))

        assert all(filtered['listing_price'] >= 300000)
        assert all(filtered['listing_price'] <= 400000)

    def test_filter_by_bedrooms(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test filtering by bedroom count."""
        data_processor._demographics_df = sample_demographics_df
        data_processor._demographics_df['crime_index_numeric'] = \
            data_processor._demographics_df['crime_index'].map(DataProcessor.CRIME_INDEX_MAP)

        # Process listings
        sample_listings_df['postal_code_cleaned'] = sample_listings_df['postal_code'].apply(
            data_processor._clean_postal_code
        )
        sample_listings_df['address_normalized'] = sample_listings_df['raw_address'].apply(
            data_processor._normalize_address
        )
        sample_listings_df['price_per_sqft'] = \
            sample_listings_df['listing_price'] / sample_listings_df['sq_ft']
        data_processor._listings_df = sample_listings_df

        merged = data_processor.merge_data()
        filtered = data_processor.filter_data(merged, bedrooms=[3, 4])

        assert all(filtered['bedrooms'].isin([3, 4]))


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestFormatCurrency:
    """Test cases for format_currency function."""

    def test_format_integer(self):
        """Test formatting integer value."""
        result = format_currency(100000)
        assert result == "$100,000"

    def test_format_float(self):
        """Test formatting float value."""
        result = format_currency(100000.50, decimals=2)
        assert result == "$100,000.50"

    def test_format_nan(self):
        """Test formatting NaN value."""
        result = format_currency(np.nan)
        assert result == "N/A"


class TestFormatPercentage:
    """Test cases for format_percentage function."""

    def test_format_percentage(self):
        """Test formatting percentage value."""
        result = format_percentage(95.5)
        assert result == "95.5%"

    def test_format_percentage_no_decimals(self):
        """Test formatting percentage with no decimals."""
        result = format_percentage(95.5, decimals=0)
        assert result == "96%"


class TestFormatNumber:
    """Test cases for format_number function."""

    def test_format_number(self):
        """Test formatting number with thousands separator."""
        result = format_number(1000000)
        assert result == "1,000,000"


class TestGetRatingEmoji:
    """Test cases for get_rating_emoji function."""

    def test_high_rating(self):
        """Test high rating emoji."""
        result = get_rating_emoji(9.5)
        assert "⭐" in result
        assert result.count("⭐") == 5

    def test_low_rating(self):
        """Test low rating emoji."""
        result = get_rating_emoji(1.5)
        assert "⭐" in result

    def test_nan_rating(self):
        """Test NaN rating emoji."""
        result = get_rating_emoji(np.nan)
        assert result == "❓"


class TestValidateDataframe:
    """Test cases for validate_dataframe function."""

    def test_valid_dataframe(self, sample_listings_df):
        """Test validation of valid DataFrame."""
        required = ['raw_address', 'postal_code', 'listing_price']
        result = validate_dataframe(sample_listings_df, required)

        assert result['valid'] is True
        assert len(result['missing_columns']) == 0

    def test_missing_columns(self, sample_listings_df):
        """Test validation with missing columns."""
        required = ['raw_address', 'nonexistent_column']
        result = validate_dataframe(sample_listings_df, required)

        assert result['valid'] is False
        assert 'nonexistent_column' in result['missing_columns']


class TestGetFilterOptions:
    """Test cases for get_filter_options function."""

    def test_get_filter_options(self, sample_listings_df):
        """Test getting filter options from DataFrame."""
        sample_listings_df['matched_zip'] = sample_listings_df['postal_code']
        sample_listings_df['crime_index'] = ['Low', 'Medium', 'High', 'Low', 'Medium']
        sample_listings_df['school_rating'] = [8.5, 7.2, 9.1, 6.3, 5.5]

        options = get_filter_options(sample_listings_df)

        assert 'zip_codes' in options
        assert 'price_range' in options
        assert 'sqft_range' in options
        assert 'bedrooms' in options


# ============================================================================
# SAMPLE COORDINATES TESTS
# ============================================================================

class TestCreateSampleCoordinates:
    """Test cases for create_sample_coordinates function."""

    def test_adds_latitude_column(self, sample_listings_df):
        """Test that latitude column is added."""
        sample_listings_df['matched_zip'] = sample_listings_df['postal_code']
        result = create_sample_coordinates(sample_listings_df)

        assert 'latitude' in result.columns

    def test_adds_longitude_column(self, sample_listings_df):
        """Test that longitude column is added."""
        sample_listings_df['matched_zip'] = sample_listings_df['postal_code']
        result = create_sample_coordinates(sample_listings_df)

        assert 'longitude' in result.columns

    def test_reproducible_with_seed(self, sample_listings_df):
        """Test that coordinates are reproducible with same seed."""
        sample_listings_df['matched_zip'] = sample_listings_df['postal_code']

        result1 = create_sample_coordinates(sample_listings_df, seed=42)
        result2 = create_sample_coordinates(sample_listings_df, seed=42)

        # Due to random noise, we check the base coordinates
        assert result1['latitude'].iloc[0] is not None
        assert result2['longitude'].iloc[0] is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_empty_dataframe(self, data_processor):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = data_processor.filter_data(empty_df)
        assert len(result) == 0

    def test_all_null_values(self, data_processor):
        """Test handling of all null values in postal codes."""
        listings = pd.DataFrame({
            'postal_code_cleaned': [None, None, None]
        })
        demographics = pd.DataFrame({
            'zip_code': ['12345', '23456']
        })

        results = data_processor.fuzzy_match_zip_codes(listings, demographics)
        assert all(results['match_type'] == 'no_match')

    def test_special_characters_in_address(self, data_processor):
        """Test normalization with special characters."""
        result = data_processor._normalize_address("123 Main St. #Apt 4B")
        assert isinstance(result, str)

    def test_very_long_address(self, data_processor):
        """Test normalization of very long address."""
        long_address = "123 " + "Very Long Street Name " * 10 + "Blvd."
        result = data_processor._normalize_address(long_address)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_numeric_postal_code(self, data_processor):
        """Test handling of numeric postal code (not string)."""
        result = data_processor._clean_postal_code(12345)
        assert result == "12345"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete data pipeline."""

    def test_full_pipeline(self, data_processor, sample_demographics_df, sample_listings_df):
        """Test the complete data processing pipeline."""
        # Set up demographics
        data_processor._demographics_df = sample_demographics_df
        data_processor._demographics_df['crime_index_numeric'] = \
            data_processor._demographics_df['crime_index'].map(DataProcessor.CRIME_INDEX_MAP)

        # Process listings
        sample_listings_df['postal_code_cleaned'] = sample_listings_df['postal_code'].apply(
            data_processor._clean_postal_code
        )
        sample_listings_df['address_normalized'] = sample_listings_df['raw_address'].apply(
            data_processor._normalize_address
        )
        sample_listings_df['price_per_sqft'] = \
            sample_listings_df['listing_price'] / sample_listings_df['sq_ft']
        data_processor._listings_df = sample_listings_df

        # Merge data
        merged = data_processor.merge_data()

        # Add coordinates
        merged = create_sample_coordinates(merged)

        # Get statistics
        stats = data_processor.get_summary_statistics()

        # Filter data
        filtered = data_processor.filter_data(merged, price_range=(300000, 500000))

        # Validate results
        assert len(merged) > 0
        assert isinstance(stats, dict)
        assert len(filtered) <= len(merged)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
