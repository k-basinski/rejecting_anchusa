# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anchusa as an
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
    mdf = pd.read_csv(f'extracted_features/temp/{f}')
    mdfs.append(mdf)


df = pd.concat(mdfs, ignore_index=True)
df.to_csv('extracted_features/all_mmse.csv', index=False)
# %%
df


# %%
df

sns.lineplot(x = 'scale', y = 'mmse', hue = 'condition', data = df)
plt.savefig('figures/mmse.png', dpi=300)
plt.show()

# %%
tot = df.groupby(['subject', 'condition'])['mmse'].sum().to_frame()
sns.swarmplot(x = 'condition', y = 'mmse', data = tot, alpha = 0.8, size=3)
sns.violinplot(x = 'condition', y = 'mmse', data = tot)
plt.savefig('figures/total_mmse.png', dpi=300)
plt.show()


# %%
win_df = pd.read_csv('extracted_features/windowed_sample_entropy_window5.csv')
win_df['time'] = win_df['sample'] * an.TR
sns.lineplot(data=win_df, x='time', y='mmse', hue='condition')
plt.ylabel('entropy')
plt.savefig('figures/wndtropy_sliding_window5.png', dpi=300)
plt.show()

# %%
win_df = pd.read_csv('extracted_features/windowed_sample_entropy_window7.csv')
sns.lineplot(data=win_df, x='sample', y='mmse', hue='condition')
plt.show()
# %%
win_df = pd.read_csv('extracted_features/windowed_sample_entropy_window10.csv')
sns.lineplot(data=win_df, x='sample', y='mmse', hue='condition')
plt.show()
# %%
win_df = pd.read_csv('extracted_features/windowed_sample_entropy_window12.csv')
sns.lineplot(data=win_df, x='sample', y='mmse', hue='condition')
plt.show()
# %%
win_df
# %%
win_df = pd.read_csv('extracted_features/windowed_entropy_by_object.csv')
win_df['time'] = win_df['sample'] * an.TR
g = sns.FacetGrid(win_df, col="object", hue='condition')
g.map(sns.lineplot, 'time', 'entropy')
plt.savefig('figures/entropy_by_object.png', dpi=300)
plt.show()