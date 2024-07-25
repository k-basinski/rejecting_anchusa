# %%
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

# conditions
conds = ['2bk', '0bk']

# objects
objects = ['body', 'faces', 'places', 'tools']


# %%
dfs = []
for s in subjects:
    print('Calculating entropy for subject {s}'.format(s=s))
    for cond in conds:
        for obj in objects:
            data_list = an.get_conds(s, "wm", [f'{cond}_{obj}'], concat=False)
            for run in range(2):
                ens = an.calculate_sliding_entropy(data_list[run], win_size=window_size)
                mdf = pd.DataFrame({
                    'subject': s,
                    'condition': cond,
                    'object': obj,
                    'run': run,
                    'entropy': ens,
                })
                dfs.append(mdf)

df = pd.concat(dfs)
df.index.name = 'sample'
df.to_csv(f'{output_folder}/windowed_entropy_by_object.csv')

