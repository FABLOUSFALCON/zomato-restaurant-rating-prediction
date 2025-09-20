from pathlib import Path
from typing import Any, Tuple, List
import re
import ast
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

from zomato_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


class ZomatoDataProcessor:
    """
    Production-ready data processor that converts raw Zomato data into clean, feature-rich format.
    Implements all the sophisticated preprocessing logic from notebook 02.
    """

    def __init__(self):
        """Initialize the processor with encoding patterns and mappings."""
        # Mojibake patterns discovered in notebook analysis
        self.encoding_patterns = {
            'Ã¢â‚¬Å"': '"',
            "Ã¢â‚¬Â": '"',
            "Ã¢â‚¬â„¢": "'",
            "Ã¢â‚¬Å¡": "'",
            "Ã¢â‚¬": "–",
            "ÃƒÂ": "Â",
            "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ¼": "¼",
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            'â€"': "–",
            "â€¦": "...",
        }

        # Known non-numeric rating strings
        self.non_numeric_ratings = [
            "NEW",
            "new",
            "New",
            "-",
            "–",
            "OPENING SOON",
            "Opening Soon",
            "Temporarily Closed",
            "TEMPORARILY CLOSED",
            "Not rated",
            "NOT RATED",
        ]

        logger.info(
            "✅ ZomatoDataProcessor initialized with encoding patterns and rating mappings"
        )

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to snake_case format.
        Based on notebook 02 standardization logic.
        """
        logger.info("🔧 Standardizing column names...")

        column_mapping = {
            "approx_cost(for two people)": "cost_for_two",
            "listed_in(city)": "area",
            "listed_in(type)": "category",
        }

        df_clean = df.rename(columns=column_mapping)

        # Convert remaining columns to snake_case
        def to_snake_case(name):
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            name = re.sub(r"_+", "_", name)
            return name.lower().strip("_")

        df_clean.columns = [to_snake_case(col) for col in df_clean.columns]

        logger.success(f"✅ Column names standardized: {list(df_clean.columns)}")
        return df_clean

    def clean_rate_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert the rate column from messy strings to proper numeric values.
        Implements the sophisticated logic from notebook 02.
        """
        logger.info("🔧 Cleaning rate column...")
        df_clean = df.copy()

        # Convert non-numeric strings to NaN
        mask = df_clean["rate"].isin(self.non_numeric_ratings)
        df_clean.loc[mask, "rate"] = np.nan
        logger.info(f"Converted {mask.sum()} non-numeric ratings to NaN")

        # Remove '/5' suffix and convert to float
        df_clean["rate"] = (
            df_clean["rate"].astype(str).str.replace("/5", "", regex=False)
        )
        df_clean["rate"] = pd.to_numeric(df_clean["rate"], errors="coerce")

        # Validate range (should be 0-5)
        invalid_range = (df_clean["rate"] < 0) | (df_clean["rate"] > 5)
        if invalid_range.any():
            logger.warning(
                f"Found {invalid_range.sum()} ratings outside 0-5 range, setting to NaN"
            )
            df_clean.loc[invalid_range, "rate"] = np.nan

        logger.success(
            f"✅ Rate column cleaned. Valid ratings: {(~df_clean['rate'].isna()).sum()}"
        )
        return df_clean

    def clean_cost_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the cost_for_two column by removing commas and converting to numeric."""
        logger.info("🔧 Cleaning cost_for_two column...")
        df_clean = df.copy()

        # Remove commas and convert to numeric
        df_clean["cost_for_two"] = (
            df_clean["cost_for_two"].astype(str).str.replace(",", "")
        )
        df_clean["cost_for_two"] = pd.to_numeric(
            df_clean["cost_for_two"], errors="coerce"
        )

        # Remove unrealistic values (< 50 or > 10000)
        unrealistic = (df_clean["cost_for_two"] < 50) | (
            df_clean["cost_for_two"] > 10000
        )
        if unrealistic.any():
            logger.info(f"Setting {unrealistic.sum()} unrealistic cost values to NaN")
            df_clean.loc[unrealistic, "cost_for_two"] = np.nan

        logger.success("✅ Cost column cleaned and converted to numeric")
        return df_clean

    def convert_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Yes/No columns to binary 1/0."""
        logger.info("🔧 Converting binary columns...")
        df_clean = df.copy()

        binary_cols = ["online_order", "book_table"]
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})
                logger.info(f"Converted {col} to binary (1/0)")

        logger.success("✅ Binary columns converted")
        return df_clean

    def fix_text_encoding(self, text: Any) -> str:
        """
        Fix mojibake encoding issues in text.
        Implements the sophisticated encoding fix logic from notebook 02.
        """
        if pd.isna(text) or not isinstance(text, str):
            return text

        # Apply all encoding pattern fixes
        fixed_text = text
        for broken, correct in self.encoding_patterns.items():
            fixed_text = fixed_text.replace(broken, correct)

        return fixed_text

    def clean_text_columns_parallel(
        self, df: pd.DataFrame, text_columns: List[str]
    ) -> pd.DataFrame:
        """
        Clean text columns in parallel using ThreadPoolExecutor.
        Implements sophisticated parallel processing for production performance.
        """
        logger.info(f"🔧 Cleaning text columns in parallel: {text_columns}")
        df_clean = df.copy()

        def process_column(col_name):
            if col_name in df_clean.columns:
                logger.info(f"Processing column: {col_name}")
                df_clean[col_name] = df_clean[col_name].apply(self.fix_text_encoding)
                return f"✅ {col_name} processed"
            return f"⚠️ {col_name} not found"

        # Process columns in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_column, col) for col in text_columns]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        for result in results:
            logger.info(result)

        logger.success("✅ All text columns cleaned in parallel")
        return df_clean

    def remove_duplicates_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates using sophisticated multi-stage strategy from notebook 02.
        """
        logger.info("🔧 Removing duplicates with advanced strategy...")
        initial_count = len(df)

        # Stage 1: Remove exact duplicates (excluding URL which might be unique)
        subset_cols = [col for col in df.columns if col not in ["url", "phone"]]
        df_clean = df.drop_duplicates(subset=subset_cols, keep="first")
        stage1_removed = initial_count - len(df_clean)

        # Stage 2: Remove near-duplicates based on name + location + cost
        if all(col in df_clean.columns for col in ["name", "location", "cost_for_two"]):
            df_clean = df_clean.drop_duplicates(
                subset=["name", "location", "cost_for_two"], keep="first"
            )
            stage2_removed = initial_count - stage1_removed - len(df_clean)
        else:
            stage2_removed = 0

        total_removed = initial_count - len(df_clean)
        logger.success(
            f"✅ Removed {total_removed} duplicates (Stage 1: {stage1_removed}, Stage 2: {stage2_removed})"
        )
        return df_clean

    def impute_missing_values_smart(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart missing value imputation using context-aware strategies from notebook 02.
        """
        logger.info("🔧 Applying smart missing value imputation...")
        df_clean = df.copy()

        # Strategy 1: Impute cost_for_two based on location and restaurant type
        if "cost_for_two" in df_clean.columns and "location" in df_clean.columns:
            cost_missing = df_clean["cost_for_two"].isna()
            if cost_missing.any():
                # Group by location and use median
                location_cost_median = df_clean.groupby("location")[
                    "cost_for_two"
                ].median()

                for location in location_cost_median.index:
                    mask = (df_clean["location"] == location) & df_clean[
                        "cost_for_two"
                    ].isna()
                    df_clean.loc[mask, "cost_for_two"] = location_cost_median[location]

                # Fill remaining with overall median
                overall_median = df_clean["cost_for_two"].median()
                df_clean["cost_for_two"] = df_clean["cost_for_two"].fillna(
                    overall_median
                )

                logger.info(
                    f"Imputed {cost_missing.sum()} missing cost values using location-based strategy"
                )

        # Strategy 2: Impute rate based on votes and cost (higher cost/votes usually mean higher rating)
        if "rate" in df_clean.columns and "votes" in df_clean.columns:
            rate_missing = df_clean["rate"].isna()
            if rate_missing.any():
                # Create bins based on votes and cost
                df_clean["votes_bin"] = pd.qcut(
                    df_clean["votes"],
                    q=5,
                    labels=["very_low", "low", "medium", "high", "very_high"],
                    duplicates="drop",
                )
                df_clean["cost_bin"] = pd.qcut(
                    df_clean["cost_for_two"],
                    q=5,
                    labels=["budget", "affordable", "moderate", "expensive", "premium"],
                    duplicates="drop",
                )

                # Impute based on similar restaurants
                for votes_bin in df_clean["votes_bin"].unique():
                    for cost_bin in df_clean["cost_bin"].unique():
                        mask = (
                            (df_clean["votes_bin"] == votes_bin)
                            & (df_clean["cost_bin"] == cost_bin)
                            & df_clean["rate"].isna()
                        )
                        if mask.any():
                            similar_restaurants = df_clean[
                                (df_clean["votes_bin"] == votes_bin)
                                & (df_clean["cost_bin"] == cost_bin)
                                & df_clean["rate"].notna()
                            ]
                            if not similar_restaurants.empty:
                                median_rate = similar_restaurants["rate"].median()
                                df_clean.loc[mask, "rate"] = median_rate

                # Fill remaining with overall median
                overall_rate_median = df_clean["rate"].median()
                df_clean["rate"] = df_clean["rate"].fillna(overall_rate_median)

                # Clean up temporary columns
                df_clean = df_clean.drop(
                    ["votes_bin", "cost_bin"], axis=1, errors="ignore"
                )

                logger.info(
                    f"Imputed {rate_missing.sum()} missing rate values using similarity-based strategy"
                )

        logger.success("✅ Smart missing value imputation completed")
        return df_clean

    def process_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete data processing pipeline.
        This is the main production pipeline that handles all preprocessing.
        """
        logger.info("🚀 Starting full data processing pipeline...")

        # Step 1: Standardize column names
        df_processed = self.standardize_column_names(df)

        # Step 2: Clean and convert data types
        df_processed = self.clean_rate_column(df_processed)
        df_processed = self.clean_cost_column(df_processed)
        df_processed = self.convert_binary_columns(df_processed)

        # Step 3: Fix text encoding issues in parallel
        text_columns = [
            "name",
            "address",
            "location",
            "rest_type",
            "cuisines",
            "dish_liked",
        ]
        existing_text_cols = [
            col for col in text_columns if col in df_processed.columns
        ]
        df_processed = self.clean_text_columns_parallel(
            df_processed, existing_text_cols
        )

        # Step 4: Remove duplicates
        df_processed = self.remove_duplicates_advanced(df_processed)

        # Step 5: Smart missing value imputation
        df_processed = self.impute_missing_values_smart(df_processed)

        # Step 6: Drop unnecessary columns for modeling
        columns_to_drop = ["url", "phone", "menu_item"]  # menu_item is mostly empty
        df_processed = df_processed.drop(columns=columns_to_drop, errors="ignore")

        logger.success(
            f"🎉 Data processing pipeline completed! Final shape: {df_processed.shape}"
        )
        return df_processed


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "zomato_raw.csv",
    output_path: Path = PROCESSED_DATA_DIR / "zomato_master_processed.parquet",
):
    """
    Main function to process raw Zomato data into clean, production-ready format.
    Uses all the sophisticated preprocessing logic from notebook 02.
    """
    logger.info(f"🔄 Processing dataset from {input_path} to {output_path}")

    # Initialize processor
    processor = ZomatoDataProcessor()

    # Load raw data
    logger.info("📖 Loading raw data...")
    df_raw = pd.read_csv(input_path)
    logger.info(f"Raw data loaded: {df_raw.shape}")

    # Process data
    df_processed = processor.process_full_pipeline(df_raw)

    # Save processed data
    logger.info(f"💾 Saving processed data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(output_path, index=False)

    # Generate summary report
    logger.info("📊 Processing Summary:")
    logger.info(f"  • Input records: {len(df_raw):,}")
    logger.info(f"  • Output records: {len(df_processed):,}")
    logger.info(f"  • Records removed: {len(df_raw) - len(df_processed):,}")
    logger.info(f"  • Missing values in rate: {df_processed['rate'].isna().sum():,}")
    logger.info(
        f"  • Missing values in cost_for_two: {df_processed['cost_for_two'].isna().sum():,}"
    )

    logger.success("🎉 Dataset processing complete!")


if __name__ == "__main__":
    app()
