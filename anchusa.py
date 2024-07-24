
import numpy as np
import matplotlib.pyplot as plt
import os, requests, tarfile
import antropy as at


# Necessary for visualization
from nilearn import plotting, datasets
from nilearn.surface import vol_to_surf

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/master/nma.mplstyle")
# The download cells will store the data in nested directories starting here:
HCP_DIR = "./DATA"
if not os.path.isdir(HCP_DIR):
  os.mkdir(HCP_DIR)

# filepaths for convinience
fpath_task = os.path.join(HCP_DIR, "hcp_task")
fpath_rest = os.path.join(HCP_DIR, "hcp_rest")

# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339

# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360

# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in sec

# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]

# Each experiment was repeated multiple times in each subject
N_RUNS_REST = 4
N_RUNS_TASK = 2

# Time series data are organized by experiment, with each experiment
# having an LR and RL (phase-encode direction) acquistion
BOLD_NAMES = [
  "rfMRI_REST1_LR", "rfMRI_REST1_RL",
  "rfMRI_REST2_LR", "rfMRI_REST2_RL",
  "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR",
  "tfMRI_WM_RL", "tfMRI_WM_LR",
  "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR",
  "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
  "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR",
  "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
  "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]

# You may want to limit the subjects used during code development.
# This will use all subjects:
subjects = range(N_SUBJECTS)

def download_data():
  """Download the HCP dataset."""

  fnames = ["hcp_rest.tgz",
            "hcp_task.tgz",
            "hcp_covariates.tgz",
            "atlas.npz"]
  urls = ["https://osf.io/bqp7m/download",
          "https://osf.io/s4h8j/download",
          "https://osf.io/x5p4g/download",
          "https://osf.io/j5kuc/download"]


  for fname, url in zip(fnames, urls):
    if not os.path.isfile(fname):
      try:
        r = requests.get(url)
      except requests.ConnectionError:
        print("!!! Failed to download data !!!")
      else:
        if r.status_code != requests.codes.ok:
          print("!!! Failed to download data !!!")
        else:
          print(f"Downloading {fname}...")
          with open(fname, "wb") as fid:
            fid.write(r.content)
          print(f"Download {fname} completed!")


  fnames = ["hcp_covariates", "hcp_rest", "hcp_task"]

  for fname in fnames:
    # open file
    path_name = os.path.join(HCP_DIR, fname)
    if not os.path.exists(path_name):
      print(f"Extracting {fname}.tgz...")
      with tarfile.open(f"{fname}.tgz") as fzip:
        fzip.extractall(HCP_DIR)
    else:
      print(f"File {fname}.tgz has already been extracted.")


def load_regions_and_atlas():
  """Load region info and parcellations.

    Args:
      None
    Returns:
      Tuple containing:
        region_info (dict): region names and network assignments
        atlas (dict): parcellation on the fsaverage5 surface and approximate MNI coordinates of each region
  """
  folder = os.path.join(HCP_DIR, "hcp_task")  # choose the data directory
  regions = np.load(os.path.join(folder, "regions.npy")).T
  region_info = dict(name=regions[0].tolist(),
                    network=regions[1],
                    myelin=regions[2].astype(float)
                    )
  with np.load(f"atlas.npz") as dobj:
    atlas = dict(**dobj)

  return region_info, atlas


def read_behavior_data(kind, output_type="pandas"):
    """Read behavior data from HCP dataset.

    Args:
        kind (str): "gambling", "language", "emotion", "relational", "social" or "wm",
        output_type (str): Sets the output dataframe format. Can be "numpy" or "pandas". Default is "pandas".
    Returns:
         wm_behavior (np.ndarray or pandas DataFrame): behavior data
    """
    fpath = os.path.join(HCP_DIR, "hcp", "behavior", f"{kind}.csv")

    if output_type == "numpy":
        wm_behavior = np.genfromtxt(fpath,
                                delimiter=",",
                                names=True,
                                dtype=None,
                                encoding="utf")

    elif output_type == "pandas":
        import pandas as pd
        wm_behavior = pd.read_csv(fpath)

    return wm_behavior


def get_image_ids(name):
  """Get the 1-based image indices for runs in a given experiment.

    Args:
      name (str) : Name of experiment ("rest" or name of task) to load
    Returns:
      run_ids (list of int) : Numeric ID for experiment image files

  """
  run_ids = [
             i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code
             ]
  if not run_ids:
    raise ValueError(f"Found no data for '{name}'")
  return run_ids


def load_timeseries(subject, name, dir,
                    runs=None, concat=True, remove_mean=True):
  """Load timeseries data for a single subject.

  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of experiment ("rest" or name of task) to load
    dir (str) : data directory
    run (None or int or list of ints): 0-based run(s) of the task to load,
      or None to load all runs.
    concat (bool) : If True, concatenate multiple runs in time
    remove_mean (bool) : If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_tp array): Array of BOLD data values

  """
  # Get the list relative 0-based index of runs to use
  if runs is None:
    runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
  elif isinstance(runs, int):
    runs = [runs]

  # Get the first (1-based) run id for this experiment
  offset = get_image_ids(name)[0]

  # Load each run's data
  bold_data = [
               load_single_timeseries(subject,
                                      offset + run,
                                      dir,
                                      remove_mean) for run in runs
               ]

  # Optionally concatenate in time
  if concat:
    bold_data = np.concatenate(bold_data, axis=-1)

  return bold_data


def load_single_timeseries(subject, bold_run, dir, remove_mean=True):
  """Load timeseries data for a single subject and single run.

  Args:
    subject (int): 0-based subject ID to load
    bold_run (int): 1-based run index, across all tasks
    dir (str) : data directory
    remove_mean (bool): If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_path = os.path.join(dir, "subjects", str(subject), "timeseries")
  bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
  ts = np.load(os.path.join(bold_path, bold_file))
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts


def load_evs(subject, name, condition, dir):
  """Load EV (explanatory variable) data for one task condition.

  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of task
    condition (str) : Name of condition
    dir (str) : data directory

  Returns
    evs (list of dicts): A dictionary with the onset, duration, and amplitude
      of the condition for each run.

  """
  evs = []
  for id in get_image_ids(name):
    task_key = BOLD_NAMES[id - 1]
    ev_file = os.path.join(dir, "subjects", str(subject), "EVs",
                           task_key, f"{condition}.txt")
    ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
    ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
    evs.append(ev)
  return evs

def condition_frames(run_evs, skip=0):
  """Identify timepoints corresponding to a given condition in each run.

  Args:
    run_evs (list of dicts) : Onset and duration of the event, per run
    skip (int) : Ignore this many frames at the start of each trial, to account
      for hemodynamic lag

  Returns:
    frames_list (list of 1D arrays): Flat arrays of frame indices, per run

  """
  frames_list = []
  for ev in run_evs:

    # Determine when trial starts, rounded down
    start = np.floor(ev["onset"] / TR).astype(int)

    # Use trial duration to determine how many frames to include for trial
    duration = np.ceil(ev["duration"] / TR).astype(int)

    # Take the range of frames that correspond to this specific trial
    frames = [s + np.arange(skip, d) for s, d in zip(start, duration)]

    frames_list.append(np.concatenate(frames))

  return frames_list


def selective_average(timeseries_data, ev, skip=0):
  """Take the temporal mean across frames for a given condition.

  Args:
    timeseries_data (array or list of arrays): n_parcel x n_tp arrays
    ev (dict or list of dicts): Condition timing information
    skip (int) : Ignore this many frames at the start of each trial, to account
      for hemodynamic lag

  Returns:
    avg_data (1D array): Data averagted across selected image frames based
    on condition timing

  """
  # Ensure that we have lists of the same length
  if not isinstance(timeseries_data, list):
    timeseries_data = [timeseries_data]
  if not isinstance(ev, list):
    ev = [ev]
  if len(timeseries_data) != len(ev):
    raise ValueError("Length of `timeseries_data` and `ev` must match.")

  # Identify the indices of relevant frames
  frames = condition_frames(ev, skip)

  # Select the frames from each image
  selected_data = []
  for run_data, run_frames in zip(timeseries_data, frames):
    run_frames = run_frames[run_frames < run_data.shape[1]]
    selected_data.append(run_data[:, run_frames])

  # Take the average in each parcel
  avg_data = np.concatenate(selected_data, axis=-1).mean(axis=-1)

  return avg_data


def seconds_to_samples(sec):
  """Converts seconds to samples (multiply by TR and floor).
  Args:
    sec (float): time in seconds
  Return:
    samples (int): time in samples
    """
  return int(sec * TR)


def get_conds(subject, task, conds, concat=False, skip=0):
    """Get timeseries for conditions of interest.
    Args:
        subject (int): subject id,
        task (string): task id ("wm", "gambling" etc.)
        conds (list): list of conditions to concatenate.
    Returns:
        (n_parcel x n_tp array): Array of concatenated BOLD data values
    """
    # load subject timeseries
    ts = load_timeseries(subject, task, fpath_task)

    cuts = []
    # for each condition
    for cond in conds:
        # load evs
        evs = load_evs(subject, task, cond, fpath_task)

        # indexing array
        frames = condition_frames(evs, skip=skip)

        # for each run
        for run in range(len(frames)):

            # cut
            ts_cut = ts[:, frames[run]]

            # append to cuts list
            cuts.append(ts_cut)


    # Optionally concatenate in time
    if concat:
        cuts = np.concatenate(cuts, axis=-1)

    return cuts


def extract_frontoparietal_parcels(two_back, no_back, region_info):
    """
    Extracts the Frontoparietal network parcels from two_back and no_back datasets.

    Parameters:
    - two_back: List of NumPy arrays, each representing the time series data for a participant
                during the two-back task with shape (regions:360, subjects:152).
    - no_back: List of NumPy arrays, each representing the time series data for a participant
                during the no-back task with shape (regions:360, subjects:152).
    - region_info: A dictionary containing region metadata, including network assignments.

    Returns:
    - two_back_fpn: List of NumPy arrays containing only the Frontoparietal network parcels
                     from the two_back dataset. (time:399 x parcels:50 x subjects:152)
    - no_back_fpn: List of NumPy arrays containing only the Frontoparietal network parcels
                    from the no_back dataset. (time x parcels x subjects)
    """
    # Identify the indices of the 'Frontopariet' network
    fpn_network_name = "Frontopariet"
    fpn_indices = [i for i, network in enumerate(region_info['network']) if network == fpn_network_name]

    two_back_fpn = []
    no_back_fpn = []

    # Extract Frontopariet parcels for two_back
    for subject_two_back in two_back:
        frontopariet_two_back = subject_two_back[fpn_indices, :]  # Assuming subject_two_back is a numpy array
        two_back_fpn.append(frontopariet_two_back)

    # Extract Frontopariet parcels for no_back
    for subject_no_back in no_back:
        frontopariet_no_back = subject_no_back[fpn_indices, :]  # Assuming subject_no_back is a numpy array
        no_back_fpn.append(frontopariet_no_back)

    return two_back_fpn, no_back_fpn


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