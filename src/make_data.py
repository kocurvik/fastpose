# -*- coding: utf-8 -*-
"""Turn fastposebench CSV results into the bench.json the page embeds.

    python src/make_data.py --csv_dir /path/to/fastposebench/csv_results

Each CSV is named <prefix>_<matcher>_<budget>_..._pose_results.csv, where the
budget is the total sampled matches for RoMa v2 and the maximum number of
keypoints for LoMa. The `cpp` column is the CPU count per process, with 0
standing for the CUDA run; those become the four backends the page compares.
"""
import argparse
import glob
import json
import os

import pandas as pd

DEVICE_BY_CPP = {0: 'fastpose_gpu', 1: 'fastpose_1c', 4: 'fastpose_4c'}

KEEP = ['matcher', 'n', 'dataset', 'problem', 'iters', 'device',
        'mean_runtime', 'pose_mAA_10', 'pose_mAA_5', 'median_pose_err', 'mean_inliers']

# The varying-focal RePoseD solver returns degenerate poses under the 4-CPU
# driver (mAA collapses, inlier rate drops to a few percent) while its 1-CPU and
# CUDA runs are normal. That is a bug rather than a measurement, so the page does
# not report it.
DROP = [('vf_reposed', 'fastpose_4c')]


def load(csv_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(csv_dir, '*_pose_results.csv'))):
        name = os.path.basename(path)
        matcher = 'RoMa' if 'roma' in name else 'LoMa'
        budget = int(next(part for part in name.split('_') if part.isdigit()))

        df = pd.read_csv(path)
        df['matcher'] = matcher
        df['n'] = budget
        df['problem'] = df['solver'].str.rsplit('_', n=1).str[0]
        backend = df['solver'].str.rsplit('_', n=1).str[-1]
        df['device'] = df['cpp'].map(DEVICE_BY_CPP)
        df.loc[backend == 'poselib', 'device'] = 'poselib_1c'
        frames.append(df)

    if not frames:
        raise SystemExit('no *_pose_results.csv found in %s' % csv_dir)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', required=True,
                    help='fastposebench csv_results directory')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'bench.json'))
    args = ap.parse_args()

    df = load(args.csv_dir)
    for problem, device in DROP:
        dropped = (df['problem'] == problem) & (df['device'] == device)
        print('dropping %d %s / %s rows' % (dropped.sum(), problem, device))
        df = df[~dropped]

    rows = df[KEEP].round(4).to_dict('records')
    with open(args.out, 'w') as f:
        json.dump({'rows': rows}, f, separators=(',', ':'))
    print('wrote %s (%d rows, %.1f KB)'
          % (args.out, len(rows), os.path.getsize(args.out) / 1024.0))


if __name__ == '__main__':
    main()
