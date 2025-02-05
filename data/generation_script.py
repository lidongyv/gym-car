"""
Encapsulate generate data to make it parallel

python data/generation_script.py --threads 1
"""
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--threads', type=int, help="Number of threads")
parser.add_argument('--policy', type=str, choices=['brown', 'white'],
                    help="Directory to store rollout directories of each thread",
                    default='brown')
args = parser.parse_args()

def _threaded_generation(i):
    tdir = join('datasets/carracing/thread_{}'.format(i))
    makedirs(tdir, exist_ok=True)
    # cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    # cmd += ['--server-num={}'.format(i + 1)]
    cmd = ["python", "-m", "data.carracing", "--dir",
            tdir,"--policy", args.policy]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


with Pool(args.threads) as p:
    p.map(_threaded_generation, range(args.threads))
