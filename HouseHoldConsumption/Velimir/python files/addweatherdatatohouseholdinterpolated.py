from pathlib import Path
import pandas as pd

weather_path = "./Sprint 4/weather-data/weather-data-household-interpolated.csv"
csv2 = pd.read_csv(weather_path, sep=",")

output_dir = Path("./Sprint 4/household-data-appliances-interpolated+weather/")
output_dir.mkdir(parents=True, exist_ok=True)

for house_num in range(7, 14):
    csv1_path = f"./Sprint 4/household-data-appliances-interpolated/CleerResampel_House_{house_num}.csv"
    csv1 = pd.read_csv(csv1_path, sep=",")

    csv1["temperature_2m"] = csv2["temperature_2m"].values
    csv1["relative_humidity_2m"] = csv2["relative_humidity_2m"].values

    csv1["Time"] = pd.to_datetime(csv1["Time"], format="%Y-%m-%d %H:%M:%S")
    csv1 = csv1.sort_values("Time")

    output_path = output_dir / f"CleerResampel_House_{house_num}.csv"
    csv1.to_csv(output_path, sep=",", index=False)
    print(f"Saved: {output_path}")