import pandas as pd

df = pd.read_csv("../data/processed/filtered_titles_final.csv", header= None)

unique = df[0].unique()

pd.DataFrame(unique, columns=["Job Title"]).to_csv("../data/processed/unique_titles_final.csv", index=False)
