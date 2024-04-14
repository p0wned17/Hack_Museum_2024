import pandas as pd

df = pd.read_json("json_file.json")

num_parts = 20

part_size = len(df) // num_parts
for i, part in enumerate(range(0, len(df), part_size)):
    df_part = df.iloc[part : part + part_size]

    df_part.to_csv(
        f"path_to_save_folder/part_{i}.csv",
        index=False,
    )

print("DataFrame был разбит на части и сохранен в папку 'parts'.")
