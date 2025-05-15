import os
import pandas as pd

# Define the target columns EXACTLY as per README.md
TARGET_COLUMNS = [
    "ID",  # -> from document_id
    "Matter",  # -> from besluit_id
    "Family",  # -> from inventory_number (or besluit_id, adjust if needed)
    "Document",  # -> from document_name
    "File Type",  # -> from file_type
    "Datum",  # -> from document_date
    "Document Link",  # -> from document_url
    "besluit_id",  # -> from besluit_id (keeping original)
    "available",  # -> calculated from file_path
]


def check_file_availability(file_path):
    """Checks if a file exists based on the provided path."""
    if pd.isna(file_path) or not isinstance(file_path, str) or not file_path.strip():
        return False
    # Assume file_path is relative to workspace root if not absolute
    absolute_path = (
        os.path.join(os.getcwd(), file_path)
        if not os.path.isabs(file_path)
        else file_path
    )
    return os.path.exists(absolute_path)


def transform_scraped_metadata(source_csv_path):
    """
    Transforms the scraped metadata CSV into the standardized VWS format.
    Reads from source_csv_path and returns a DataFrame in the target format.
    """
    # Check if necessary columns exist in scraped_df
    required_source_cols = [
        "document_id",
        "besluit_id",
        "inventory_number",  # Used for Family column
        "document_name",
        "file_type",
        "document_date",
        "document_url",
        "file_path",  # Used for 'available' column
    ]

    try:
        # Read source CSV, ensuring key IDs are strings
        df = pd.read_csv(
            source_csv_path,
            dtype={"document_id": str, "besluit_id": str, "inventory_number": str},
        )
        print(f"Successfully loaded source data from: {source_csv_path}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Source CSV file not found at {source_csv_path}")
        return None
    except Exception as e:
        print(f"FATAL ERROR reading source CSV ({source_csv_path}): {str(e)}")
        return None

    # Verify required columns are present
    missing_cols = [col for col in required_source_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Error: Missing required columns in source CSV ({source_csv_path}): {missing_cols}"
        )
        return None

    # Create a new DataFrame for the target format
    target_df = pd.DataFrame()

    # --- Column Mapping ---
    target_df["ID"] = df["document_id"]
    target_df["Matter"] = df["besluit_id"]
    # Decide on 'Family': Using inventory_number for now, change to 'besluit_id' if preferred
    target_df["Family"] = df["document_id"]
    target_df["Document"] = df["document_name"]
    target_df["File Type"] = df["file_type"]
    target_df["Datum"] = df["document_date"]
    target_df["Document Link"] = df["document_url"]
    target_df["besluit_id"] = df["besluit_id"]  # Keep the original column

    # Calculate 'available' column
    target_df["available"] = df["file_path"].apply(check_file_availability)

    # Ensure the final DataFrame has exactly the TARGET_COLUMNS in the correct order
    target_df = target_df[TARGET_COLUMNS]

    print(f"Transformation complete. {len(target_df)} records processed.")
    return target_df


if __name__ == "__main__":

    # Define path for the source scraped data
    source_csv_path = "data/woo_scraped/documents_metadata.csv"

    # Define output path for the standardized file
    output_path = "data/metadata/vws_metadata_standardized.csv"
    output_folder = os.path.dirname(output_path)

    # Perform the transformation
    final_df = transform_scraped_metadata(source_csv_path)

    if final_df is not None:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                print(f"Created output directory: {output_folder}")
            except OSError as e:
                print(f"Error creating output directory {output_folder}: {e}")
                final_df = None  # Prevent saving if directory fails

    # Save results to the standardized CSV if transformation was successful
    if final_df is not None:
        if not final_df.empty:
            try:
                final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                print(
                    f"\nStandardized VWS metadata saved successfully to: {output_path}"
                )
            except Exception as e:
                print(f"Error saving final CSV to {output_path}: {str(e)}")
        else:
            print("\nTransformed DataFrame is empty. No data saved.")

        # Display sample results if the dataframe is not empty
        if not final_df.empty:
            print("\nSample of final standardized documents:")
            print(final_df.head())
        else:
            print("\nTransformed DataFrame is empty. No sample data to display.")
    else:
        print("\nScript finished with errors. No output file generated.")
