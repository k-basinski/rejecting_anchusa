import numpy as np
from sys import argv
import pandas as pd
import anchusa as an
import antropy as at



# CONFIG

# You may want to limit the subjects used during code development.
N_SUBJECTS = 339

subjects = range(N_SUBJECTS)

region_info, atlas = an.load_regions_and_atlas()

start, stop = 0, 339

output_folder = 'extracted_features'

window_size = 5

# list all 2-back conditions
two_back_conds = ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools']


# list all 0-back conditions
no_back_conds = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']



# load data
two_back, no_back = [], []

for s in subjects:
    two_backs_per_sub = an.get_conds(s, "wm", two_back_conds, concat=False)
    no_backs_per_sub = an.get_conds(s, "wm", no_back_conds, concat=False)
    two_back.append(two_backs_per_sub)
    no_back.append(no_backs_per_sub)

## Uses the extract_frontoparietal_parcels to extract timeseries of parcels belonging to FPN
two_back_fpn, no_back_fpn = an.extract_frontoparietal_parcels(two_back, no_back, region_info)

# %%
dfs = []

s = 0

ens = an.calculate_sliding_entropy(two_back_fpn[s], win_size=window_size)

# %%
for s in subjects[start:stop]:
    print(f'Calculating sliding window entropy for subject: {s}...', end='')
    res = an.calculate_sliding_entropy(two_back_fpn[s], win_size=window_size)
    mdf = pd.DataFrame({
        'subject': s,
        'condition': 'two_back',
        'mmse': res,
    })
    dfs.append(mdf)

    res = an.calculate_sliding_entropy(no_back_fpn[s], win_size=window_size)
    mdf = pd.DataFrame({
        'subject': s,
        'condition': 'no_back',
        'mmse': res,
    })

    dfs.append(mdf)
    print('...done.')
# %%


# %%
df = pd.concat(dfs)
df.index.name = 'sample'
df.to_csv(f'{output_folder}/windowed_sample_entropy_window{window_size}.csv')

