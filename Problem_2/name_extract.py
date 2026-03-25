import pandas as pd
import random
import requests

# Direct raw URLs 
MALE_URL = "https://gist.githubusercontent.com/mbejda/7f86ca901fe41bc14a63/raw/Indian-Male-Names.csv"
FEMALE_URL = "https://gist.githubusercontent.com/mbejda/9b93c7545c9dd93060bd/raw/Indian-Female-Names.csv"

male_df = pd.read_csv(MALE_URL)
female_df = pd.read_csv(FEMALE_URL)

# Extract names, clean, and combine
all_names = pd.concat([
    male_df['name'].dropna(),
    female_df['name'].dropna()
]).str.strip().str.title().unique().tolist()

print(f"Total unique Indian names available: {len(all_names)}")

# Select exactly 1000 random names
random.seed(42)   
selected_names = random.sample(all_names, 1000)

# Save only the final file
with open('Problem_2/TrainingNames.txt', 'w', encoding='utf-8') as f:
    for name in selected_names:
        f.write(name + '\n')
