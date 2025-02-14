import pandas as pd

# Load the datasets
df_cannon = pd.read_csv("matrix_performance_cannon.csv")
df_cannon["algorithm"] = "Cannon"

df_row = pd.read_csv("matrix_performance_row.csv")
df_row["algorithm"] = "Row"

df_fox = pd.read_csv("matrix_performance_fox.csv")
df_fox["algorithm"] = "Fox"

# Combine all dataframes
df_combined = pd.concat([df_cannon, df_row, df_fox])

# Save the combined file
df_combined.to_csv("matrix_performance_combined.csv", index=False)

print("Combined CSV file saved successfully!")
