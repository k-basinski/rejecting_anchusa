# %%
import numpy as np
from sys import argv
import pandas as pd
import anchusa as an
import antropy as at

# %%


def calculate_entropy(data):
    en = at.sample_entropy(data.reshape(-1))
    return en


def sliding_window(data, win_size):
    return np.lib.stride_tricks.sliding_window_view(data, win_size, axis=1)

def calculate_sliding_entropy(data, win_size=10):
    # calculate sliding window mmse
    win = sliding_window(data, win_size=win_size)
    # init result array
    entropies = []
    # iterate over all windows
    for step in range(win.shape[1]):
        re = calculate_entropy(win[:, step, :])
        entropies.append(re)

    return entropies

# CONFIG

# You may want to limit the subjects used during code development.
N_SUBJECTS = 339

subjects = range(N_SUBJECTS)

region_info, atlas = an.load_regions_and_atlas()

start, stop = 0, 339

output_folder = 'extracted_features'

window_size = 10

# list all 2-back conditions
two_back_conds = ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools']


# list all 0-back conditions
no_back_conds = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']


# %%
# load data
two_back, no_back = [], []

for s in subjects:
    two_backs_per_sub = an.get_conds(s, "wm", two_back_conds, concat=True)
    no_backs_per_sub = an.get_conds(s, "wm", no_back_conds, concat=True)
    two_back.append(two_backs_per_sub)
    no_back.append(no_backs_per_sub)

## Uses the extract_frontoparietal_parcels to extract timeseries of parcels belonging to FPN
two_back_fpn, no_back_fpn = an.extract_frontoparietal_parcels(two_back, no_back, region_info)

# %%
dfs = []

s = 0

ens = calculate_sliding_entropy(two_back_fpn[s], win_size=window_size)

# %%
for s in subjects[start:stop]:
    print(f'Calculating sliding window entropy for subject: {s}...', end='')
    res = calculate_sliding_entropy(two_back_fpn[s], win_size=window_size)
    mdf = pd.DataFrame({
        'subject': s,
        'condition': 'two_back',
        'mmse': res,
    })
    dfs.append(mdf)

    res = calculate_sliding_entropy(no_back_fpn[s], win_size=window_size)
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

df.to_csv(f'{output_folder}/windowed_sample_entropy_window{window_size}.csv')

