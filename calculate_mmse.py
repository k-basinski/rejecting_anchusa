# %%
import numpy as np
from sys import argv
import EntropyHub as eh
import pandas as pd
import anchusa as an


# %%


def calculate_mmse(data, scales=4, r_param=1.75, plotx=False):
    # make the MMSE object
    mmse_obj = eh.MSobject('MvFuzzEn', Fx='constgaussian', r=r_param, Norm=True)
    msx, cix = eh.cMvMSEn(data, mmse_obj, Scales=scales, Plotx=plotx)
    return msx


def sliding_window(data, win_size):
    return np.lib.stride_tricks.sliding_window_view(data, win_size, axis=1)


def calculate_sliding_mmse(data, win_size=5, scales=4, r_param=1.75):
    # calculate sliding window mmse
    win = sliding_window(data, win_size=5)
    # init result array
    mmse_win = np.zeros((scales, win.shape[1]))
    # iterate over all windows
    for step in range(win.shape[1]):
        mmse_win[:, step] = calculate_mmse(win[:, step, :], scales=scales)


# You may want to limit the subjects used during code development.
N_SUBJECTS = 339

subjects = range(N_SUBJECTS)

region_info, atlas = an.load_regions_and_atlas()

# start, stop = 0, 1
start, stop = int(argv[1]), int(argv[2])
r_param = 1.75

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
# calculate mmse for two_back
scales = 8

dfs = []

for s in subjects[start:stop]:
    print(f'Calculating mmse for subject: {s}...', end='')
    res = calculate_mmse(two_back_fpn[s].T, scales=scales)
    mdf = pd.DataFrame({
        'subject': s,
        'condition': 'two_back',
        'scale': np.array(range(scales)),
        'mmse': res,
    })
    dfs.append(mdf)

    res = calculate_mmse(no_back_fpn[s].T, scales=scales)
    mdf = pd.DataFrame({
        'subject': s,
        'condition': 'no_back',
        'scale': np.array(range(scales)),
        'mmse': res,
    })
    dfs.append(mdf)
    print('...done.')

# %%

df = pd.concat(dfs)



df.to_csv(f'extracted_features/mmse_{start}_{stop}.csv')
