import pandas as pd
import re
from pathlib import Path

# -----------------------------
# FOLDERS / FILES
# -----------------------------
csv_folder = Path("./Sprint 4/household-data-appliances-interpolated+weather")
metadata_path = Path("./Sprint 4/weather-data/MetaData_Tables (1).xlsx")

output_folder = Path("./Sprint 4/add-metadata-to-houses")
output_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# READ METADATA ONCE
# -----------------------------
metadata = pd.read_excel(
    metadata_path,
    sheet_name="Sheet1",
    header=1
)

metadata.columns = metadata.columns.astype(str).str.strip()

for col in metadata.columns:
    if metadata[col].dtype == "object":
        metadata[col] = metadata[col].astype(str).str.replace("\t", "", regex=False).str.strip()

metadata = metadata.dropna(subset=["House"])
metadata["House"] = metadata["House"].astype(int)

# -----------------------------
# LOOP THROUGH ALL HOUSE CSV FILES
# -----------------------------
csv_files = sorted(csv_folder.glob("CleerResampel_House_*.csv"))

if not csv_files:
    raise FileNotFoundError("No house CSV files found in the folder.")

for csv_path in csv_files:
    print(f"Processing: {csv_path.name}")

    # Get house number from filename
    match = re.search(r"House[_\s]*(\d+)", csv_path.stem, re.IGNORECASE)

    if not match:
        print(f"Skipped: could not find house number in {csv_path.name}")
        continue

    house_number = int(match.group(1))

    # Find metadata row for this house
    house_meta = metadata[metadata["House"] == house_number]

    if house_meta.empty:
        print(f"Skipped: no metadata found for House {house_number}")
        continue

    house_meta = house_meta.iloc[0]

    # Read CSV
    df = pd.read_csv(csv_path)

    # Add metadata
    df["Occupancy"] = int(house_meta["Occupancy"])
    df["Appliances Owned"] = int(house_meta["Appliances Owned"])

    house_type = str(house_meta["Type"]).strip()

    df["Detached"] = 1 if house_type == "Detached" else 0
    df["Semi-detached"] = 1 if house_type == "Semi-detached" else 0
    df["Mid-terrace"] = 1 if house_type == "Mid-terrace" else 0

    df["Size"] = int(str(house_meta["Size"]).split()[0])

    # Save result
    output_path = output_folder / f"{csv_path.stem}_with_metadata.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

print("All done.")