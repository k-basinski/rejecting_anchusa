# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
res_files = [
    'mmse_0_50.csv',
    'mmse_50_100.csv',
    'mmse_100_150.csv',
    'mmse_150_200.csv',
    'mmse_200_250.csv',
    'mmse_250_300.csv',
    'mmse_300_339.csv',
]

mdfs = []
for f in res_files:
    mdf = pd.read_csv(f'extracted_features/{f}')
    mdfs.append(mdf)


df = pd.concat(mdfs, ignore_index=True)
df.to_csv('extracted_features/all_mmse.csv', index=False)
# %%
df


# %%
df

sns.lineplot(x = 'scale', y = 'mmse', hue = 'condition', data = df)

plt.show()
# %%
# sns.stripplot(x = 'scale', y = 'mmse', hue = 'condition', data = df)
sns.boxplot(x = 'scale', y = 'mmse', hue = 'condition', data = df)
plt.show()

# %%
win_df = pd.read_csv('extracted_features/windowed_sample_entropy.csv')
sns.lineplot(data=win_df, x='sample', y='mmse', hue='condition')
plt.show()
