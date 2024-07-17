import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/h3cstore_nt/pc_embedding/mm3d/LMSYS/data/train.csv")

n_dups = df.duplicated(subset=["prompt", "response_a", "response_b"], keep=False).sum()
print(f"There exist {n_dups} duplicated rows.")
df = df.drop_duplicates(
    subset=["prompt", "response_a", "response_b"], keep="first", ignore_index=True
)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"training data size: {len(train_df)}")
print(f"Val data size: {len(val_df)}")
print(f"Test data size: {len(test_df)}")

train_df.to_csv('/h3cstore_nt/pc_embedding/mm3d/LMSYS/data/split/train.csv', index=False)
val_df.to_csv('/h3cstore_nt/pc_embedding/mm3d/LMSYS/data/split/val.csv', index=False)
test_df.to_csv('/h3cstore_nt/pc_embedding/mm3d/LMSYS/data/split/test.csv', index=False)
