import json
import os

import numpy as np
import pandas as pd

# File paths
inventaris_folder = "data/woo_scraped/inventory"
json_files_folder = "standard_data_format/output_scraped/json_files_corrected_dates"

df = pd.DataFrame()
for file in os.listdir(inventaris_folder):
    if file.endswith(".csv"):
        df_new = pd.read_csv(f"{inventaris_folder}/{file}")
        df = pd.concat([df, df_new])
    if file.endswith(".xlsx"):
        df_new = pd.read_excel(f"{inventaris_folder}/{file}")
        df = pd.concat([df, df_new])
df["vws_id"] = df["Document naam"].str.split("-").str[2:].str.join("-")
print(df.head())
print(df.columns)

# Replace empty strings with NaN for uniformity
df.replace("", np.nan, inplace=True)

# Create a new column 'link' that takes the first non-missing value from the specified columns
df["link"] = (
    df[["Locatie open.minvws.nl", "Locatie document ID", "Definitief ID"]]
    .bfill(axis=1)
    .iloc[:, 0]
)

# Count the number of rows where 'link' is NaN (indicating missing values in all of the columns)
missing_count = df["link"].isna().sum()
missing_percentage = missing_count / len(df)

print(f"Total missing in 'link' column: {missing_count}")
print(f"Percentage of missing in 'link' column: {missing_percentage}")

# Add counters for logging
total_files = 0
files_with_matching_ids = 0
different_links_found = 0
updates_made = 0

json_files = os.listdir(json_files_folder)

for file in json_files:
    if file.endswith(".json"):
        total_files += 1
        with open(f"{json_files_folder}/{file}", "r") as f:
            data = json.load(f)
            id = data["vws_id"]
            json_link = data["link"]

            if id in df["vws_id"].values:
                files_with_matching_ids += 1
                df_link = df[df["vws_id"] == id]["link"].values[0]

                if pd.notna(df_link) and df_link != json_link:
                    different_links_found += 1
                    print(f"\nProcessing file: {file}")
                    print(f"VWS ID: {id}")
                    print(f"Current JSON link: {json_link}")
                    print(f"New DataFrame link: {df_link}")

                    # Update the JSON file with the new link from DataFrame
                    data["link"] = df_link
                    updates_made += 1

                    with open(f"{json_files_folder}/{file}", "w") as f:
                        json.dump(data, f, indent=4)

# Print summary statistics
print("\nSummary:")
print(f"Total JSON files processed: {total_files}")
print(f"Files with matching IDs: {files_with_matching_ids}")
print(f"Different links found: {different_links_found}")
print(f"Updates made: {updates_made}")
