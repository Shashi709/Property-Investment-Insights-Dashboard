"""
Data Processing Module for Property Investment Insights Dashboard
Author: Shashi Raj

This module handles:
- Data loading and validation
- Address normalization and cleaning
- Fuzzy matching with Levenshtein distance
- Data merging with demographic information
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


class DataProcessor:
    """
    Advanced data processing class for property listings and demographics.
    Implements fuzzy matching and robust data cleaning pipelines.
    """

    # Address normalization mappings
    ADDRESS_ABBREVIATIONS = {
        r'\bstreet\b': 'ST',
        r'\bst\.?\b': 'ST',
        r'\bavenue\b': 'AVE',
        r'\bave\.?\b': 'AVE',
        r'\bboulevard\b': 'BLVD',
        r'\bblvd\.?\b': 'BLVD',
        r'\bdrive\b': 'DR',
        r'\bdr\.?\b': 'DR',
        r'\broad\b': 'RD',
        r'\brd\.?\b': 'RD',
        r'\blane\b': 'LN',
        r'\bln\.?\b': 'LN',
        r'\bcourt\b': 'CT',
        r'\bct\.?\b': 'CT',
        r'\bplace\b': 'PL',
        r'\bpl\.?\b': 'PL',
        r'\bcircle\b': 'CIR',
        r'\bcir\.?\b': 'CIR',
        r'\bway\b': 'WAY',
        r'\bterrace\b': 'TER',
        r'\bter\.?\b': 'TER',
        r'\bparkway\b': 'PKWY',
        r'\bpkwy\.?\b': 'PKWY',
        r'\bhighway\b': 'HWY',
        r'\bhwy\.?\b': 'HWY',
        r'\bsquares?\b': 'SQ',
        r'\bsq\.?\b': 'SQ',
        r'\broute\b': 'RTE',
        r'\brte\.?\b': 'RTE',
        r'\bbypass\b': 'BYP',
        r'\bbyp\.?\b': 'BYP',
        r'\bfort\b': 'FT',
        r'\bft\.?\b': 'FT',
        r'\blight\b': 'LT',
        r'\blt\.?\b': 'LT',
        r'\bstream\b': 'STRM',
        r'\bvillage\b': 'VLG',
        r'\bvlg\.?\b': 'VLG',
        r'\bville\b': 'VL',
        r'\bvl\.?\b': 'VL',
        r'\bport\b': 'PRT',
        r'\bprt\.?\b': 'PRT',
        r'\bdam\b': 'DAM',
        r'\bknoll\b': 'KNL',
        r'\bknl\.?\b': 'KNL',
        r'\brow\b': 'ROW',
        r'\bparks?\b': 'PRK',
        r'\bprk\.?\b': 'PRK',
        r'\bkey\b': 'KEY',
        r'\bjunctions?\b': 'JCT',
        r'\bjct\.?\b': 'JCT',
        r'\bwalk\b': 'WALK',
        r'\bradial\b': 'RADL',
        r'\bradl\.?\b': 'RADL',
        r'\btrack\b': 'TRAK',
        r'\btrak\.?\b': 'TRAK',
        r'\bextension\b': 'EXT',
        r'\bext\.?\b': 'EXT',
        r'\bcircles?\b': 'CIR',
        r'\bforks?\b': 'FRK',
        r'\bfrk\.?\b': 'FRK',
        r'\bstravenue\b': 'STRA',
        r'\bstra\.?\b': 'STRA',
    }

    # Crime index mapping for numerical analysis
    CRIME_INDEX_MAP = {
        'Low': 1,
        'Medium': 2,
        'High': 3
    }

    def __init__(self, fuzzy_threshold: int = 80):
        """
        Initialize the DataProcessor.

        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self._demographics_df: Optional[pd.DataFrame] = None
        self._listings_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None

    def load_demographics(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate demographics data.

        Args:
            filepath: Path to demographics CSV file

        Returns:
            Cleaned demographics DataFrame
        """
        try:
            df = pd.read_csv(filepath, dtype={'zip_code': str})

            # Validate required columns
            required_cols = ['zip_code', 'median_income', 'school_rating', 'crime_index']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Clean and standardize ZIP codes
            df['zip_code'] = df['zip_code'].astype(str).str.strip().str.zfill(5)

            # Convert crime index to numerical
            df['crime_index_numeric'] = df['crime_index'].map(self.CRIME_INDEX_MAP)

            # Handle missing values
            df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
            df['school_rating'] = pd.to_numeric(df['school_rating'], errors='coerce')

            # Fill missing values with median (pandas 3.0 compatible)
            df['median_income'] = df['median_income'].fillna(df['median_income'].median())
            df['school_rating'] = df['school_rating'].fillna(df['school_rating'].median())
            df['crime_index_numeric'] = df['crime_index_numeric'].fillna(2)  # Default to Medium

            self._demographics_df = df
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Demographics file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading demographics data: {str(e)}")

    def load_listings(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate listings data.

        Args:
            filepath: Path to listings CSV file

        Returns:
            Cleaned listings DataFrame
        """
        try:
            df = pd.read_csv(filepath, dtype={'postal_code': str})

            # Validate required columns
            required_cols = ['raw_address', 'postal_code', 'sq_ft', 'listing_price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Clean postal codes
            df['postal_code_cleaned'] = df['postal_code'].apply(self._clean_postal_code)

            # Normalize addresses
            df['address_normalized'] = df['raw_address'].apply(self._normalize_address)

            # Clean numeric fields
            df['sq_ft'] = pd.to_numeric(df['sq_ft'], errors='coerce')
            df['listing_price'] = pd.to_numeric(df['listing_price'], errors='coerce')
            df['bedrooms'] = pd.to_numeric(df.get('bedrooms', 0), errors='coerce').fillna(0).astype(int)

            # Calculate derived metrics
            df['price_per_sqft'] = np.where(
                df['sq_ft'] > 0,
                df['listing_price'] / df['sq_ft'],
                np.nan
            )

            # Handle missing values (pandas 3.0 compatible)
            df['sq_ft'] = df['sq_ft'].fillna(df['sq_ft'].median())
            df['listing_price'] = df['listing_price'].fillna(df['listing_price'].median())

            self._listings_df = df
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Listings file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading listings data: {str(e)}")

    def _clean_postal_code(self, postal_code: str) -> Optional[str]:
        """
        Clean and validate postal codes, handling messy formats.

        Args:
            postal_code: Raw postal code string

        Returns:
            Cleaned 5-digit ZIP code or None if invalid
        """
        if pd.isna(postal_code):
            return None

        # Convert to string and strip whitespace
        code = str(postal_code).strip()

        # Handle 'XX' placeholder patterns (e.g., '325XX', '150XX')
        if 'XX' in code.upper() or 'xx' in code:
            # Extract the valid prefix
            code = re.sub(r'[Xx]+', '', code)
            if len(code) >= 3:
                # Mark as partial match needed
                return code.zfill(5)[:3] + 'XX'
            return None

        # Remove non-numeric characters
        code = re.sub(r'[^\d]', '', code)

        # Validate length
        if len(code) == 5:
            return code
        elif len(code) > 5:
            return code[:5]
        elif len(code) >= 3:
            return code.zfill(5)

        return None

    def _normalize_address(self, address: str) -> str:
        """
        Normalize address string for consistent matching.

        Args:
            address: Raw address string

        Returns:
            Normalized address string
        """
        if pd.isna(address):
            return ""

        # Convert to uppercase
        normalized = str(address).upper().strip()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove punctuation except periods
        normalized = re.sub(r'[^\w\s.]', '', normalized)

        # Apply abbreviation mappings
        for pattern, replacement in self.ADDRESS_ABBREVIATIONS.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Remove duplicate type words (e.g., "Avenue Ave" -> "AVE")
        normalized = re.sub(r'\b(ST|AVE|BLVD|DR|RD|LN|CT|PL|CIR|WAY|TER)\s+\1\b', r'\1', normalized)

        # Clean up final whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def fuzzy_match_zip_codes(
        self,
        listings_df: pd.DataFrame,
        demographics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform fuzzy matching between listings and demographics ZIP codes.

        Uses Levenshtein distance for advanced matching of messy postal codes.

        Args:
            listings_df: Listings DataFrame with postal_code_cleaned column
            demographics_df: Demographics DataFrame with zip_code column

        Returns:
            DataFrame with matched ZIP codes and match scores
        """
        valid_zip_codes = demographics_df['zip_code'].unique().tolist()
        match_results = []

        for idx, row in listings_df.iterrows():
            postal_code = row['postal_code_cleaned']

            if pd.isna(postal_code) or postal_code is None:
                match_results.append({
                    'original_index': idx,
                    'matched_zip': None,
                    'match_score': 0,
                    'match_type': 'no_match'
                })
                continue

            # Handle partial ZIP codes (with XX)
            if 'XX' in str(postal_code):
                prefix = str(postal_code).replace('XX', '')
                matching_zips = [z for z in valid_zip_codes if z.startswith(prefix)]
                if matching_zips:
                    # Use the first match for partial codes
                    match_results.append({
                        'original_index': idx,
                        'matched_zip': matching_zips[0],
                        'match_score': 85,  # Partial match confidence
                        'match_type': 'partial'
                    })
                    continue

            # Exact match first
            if postal_code in valid_zip_codes:
                match_results.append({
                    'original_index': idx,
                    'matched_zip': postal_code,
                    'match_score': 100,
                    'match_type': 'exact'
                })
                continue

            # Fuzzy match using RapidFuzz
            best_match = process.extractOne(
                postal_code,
                valid_zip_codes,
                scorer=fuzz.ratio
            )

            if best_match and best_match[1] >= self.fuzzy_threshold:
                match_results.append({
                    'original_index': idx,
                    'matched_zip': best_match[0],
                    'match_score': best_match[1],
                    'match_type': 'fuzzy'
                })
            else:
                match_results.append({
                    'original_index': idx,
                    'matched_zip': None,
                    'match_score': best_match[1] if best_match else 0,
                    'match_type': 'no_match'
                })

        return pd.DataFrame(match_results)

    def merge_data(
        self,
        listings_df: Optional[pd.DataFrame] = None,
        demographics_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge listings with demographics data using fuzzy matching.

        Args:
            listings_df: Optional listings DataFrame (uses cached if not provided)
            demographics_df: Optional demographics DataFrame (uses cached if not provided)

        Returns:
            Merged DataFrame with all property and demographic information
        """
        listings = listings_df if listings_df is not None else self._listings_df
        demographics = demographics_df if demographics_df is not None else self._demographics_df

        if listings is None or demographics is None:
            raise ValueError("Both listings and demographics data must be loaded first")

        # Perform fuzzy matching
        match_results = self.fuzzy_match_zip_codes(listings, demographics)

        # Add match results to listings
        listings_with_matches = listings.copy()
        listings_with_matches['matched_zip'] = match_results['matched_zip'].values
        listings_with_matches['match_score'] = match_results['match_score'].values
        listings_with_matches['match_type'] = match_results['match_type'].values

        # Merge with demographics
        merged_df = listings_with_matches.merge(
            demographics,
            left_on='matched_zip',
            right_on='zip_code',
            how='left'
        )

        # Calculate additional metrics
        merged_df['price_to_income_ratio'] = np.where(
            merged_df['median_income'] > 0,
            merged_df['listing_price'] / merged_df['median_income'],
            np.nan
        )

        merged_df['investment_score'] = self._calculate_investment_score(merged_df)

        self._merged_df = merged_df
        return merged_df

    def _calculate_investment_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate a composite investment score based on multiple factors.

        Score components:
        - Price per sqft (lower is better)
        - School rating (higher is better)
        - Crime index (lower is better)
        - Price to income ratio (lower is better)

        Args:
            df: Merged DataFrame

        Returns:
            Series of investment scores (0-100)
        """
        scores = pd.Series(index=df.index, dtype=float)

        # Normalize each component to 0-100 scale
        if df['price_per_sqft'].notna().any():
            price_score = 100 - (
                (df['price_per_sqft'] - df['price_per_sqft'].min()) /
                (df['price_per_sqft'].max() - df['price_per_sqft'].min() + 1) * 100
            )
        else:
            price_score = 50

        if df['school_rating'].notna().any():
            school_score = (df['school_rating'] / 10) * 100
        else:
            school_score = 50

        if df['crime_index_numeric'].notna().any():
            crime_score = (4 - df['crime_index_numeric']) / 3 * 100
        else:
            crime_score = 50

        if df['price_to_income_ratio'].notna().any():
            income_score = 100 - (
                (df['price_to_income_ratio'] - df['price_to_income_ratio'].min()) /
                (df['price_to_income_ratio'].max() - df['price_to_income_ratio'].min() + 1) * 100
            )
        else:
            income_score = 50

        # Weighted average
        weights = {
            'price': 0.25,
            'school': 0.30,
            'crime': 0.25,
            'income': 0.20
        }

        scores = (
            price_score * weights['price'] +
            school_score * weights['school'] +
            crime_score * weights['crime'] +
            income_score * weights['income']
        )

        return scores.clip(0, 100).round(2)

    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate summary statistics for the merged dataset.

        Args:
            df: Optional DataFrame (uses cached merged_df if not provided)

        Returns:
            Dictionary of summary statistics
        """
        data = df if df is not None else self._merged_df

        if data is None:
            return {}

        stats = {
            'total_listings': len(data),
            'matched_listings': len(data[data['match_type'] != 'no_match']),
            'match_rate': len(data[data['match_type'] != 'no_match']) / len(data) * 100,
            'avg_listing_price': data['listing_price'].mean(),
            'median_listing_price': data['listing_price'].median(),
            'avg_price_per_sqft': data['price_per_sqft'].mean(),
            'avg_sq_ft': data['sq_ft'].mean(),
            'total_bedrooms': data['bedrooms'].sum(),
            'avg_school_rating': data['school_rating'].mean(),
            'avg_investment_score': data['investment_score'].mean(),
            'unique_zip_codes': data['matched_zip'].nunique(),
            'price_range': {
                'min': data['listing_price'].min(),
                'max': data['listing_price'].max()
            },
            'sqft_range': {
                'min': data['sq_ft'].min(),
                'max': data['sq_ft'].max()
            }
        }

        return stats

    def filter_data(
        self,
        df: Optional[pd.DataFrame] = None,
        zip_codes: Optional[List[str]] = None,
        price_range: Optional[Tuple[float, float]] = None,
        sqft_range: Optional[Tuple[float, float]] = None,
        bedrooms: Optional[List[int]] = None,
        school_rating_min: Optional[float] = None,
        crime_index: Optional[List[str]] = None,
        investment_score_min: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Apply multiple filters to the dataset.

        Args:
            df: Optional DataFrame to filter
            zip_codes: List of ZIP codes to include
            price_range: Tuple of (min_price, max_price)
            sqft_range: Tuple of (min_sqft, max_sqft)
            bedrooms: List of bedroom counts to include
            school_rating_min: Minimum school rating
            crime_index: List of crime index levels to include
            investment_score_min: Minimum investment score

        Returns:
            Filtered DataFrame
        """
        data = df.copy() if df is not None else self._merged_df.copy()

        if data is None:
            return pd.DataFrame()

        # Apply filters
        if zip_codes:
            data = data[data['matched_zip'].isin(zip_codes)]

        if price_range:
            data = data[
                (data['listing_price'] >= price_range[0]) &
                (data['listing_price'] <= price_range[1])
            ]

        if sqft_range:
            data = data[
                (data['sq_ft'] >= sqft_range[0]) &
                (data['sq_ft'] <= sqft_range[1])
            ]

        if bedrooms:
            data = data[data['bedrooms'].isin(bedrooms)]

        if school_rating_min is not None:
            data = data[data['school_rating'] >= school_rating_min]

        if crime_index:
            data = data[data['crime_index'].isin(crime_index)]

        if investment_score_min is not None:
            data = data[data['investment_score'] >= investment_score_min]

        return data


def create_sample_coordinates(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate sample coordinates for mapping based on ZIP codes.
    In production, this would use a geocoding service.

    Args:
        df: DataFrame with ZIP codes
        seed: Random seed for reproducibility

    Returns:
        DataFrame with latitude and longitude columns
    """
    np.random.seed(seed)

    df = df.copy()

    # Generate pseudo-coordinates based on ZIP code hash
    df['latitude'] = df['matched_zip'].apply(
        lambda x: 37.0 + (hash(str(x)) % 1000) / 1000 * 5 if pd.notna(x) else np.nan
    )
    df['longitude'] = df['matched_zip'].apply(
        lambda x: -122.0 + (hash(str(x)[::-1]) % 1000) / 1000 * 5 if pd.notna(x) else np.nan
    )

    # Add some random noise for visual separation
    df['latitude'] += np.random.uniform(-0.1, 0.1, len(df))
    df['longitude'] += np.random.uniform(-0.1, 0.1, len(df))

    return df
