"""Build the pinned public Linux runtime; no registry account is required."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = 'docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755'
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-path', type=Path, help='Host volume containing Docker storage, for free-space monitoring')
    args = parser.parse_args()
    storage = args.storage_path or (Path('C:/') if Path('C:/').exists() else ROOT)
    floor = 5 * 1024**3
    cached = subprocess.run(['docker', 'image', 'inspect', BASE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    headroom = (2 if cached else 12) * 1024**3
    if shutil.disk_usage(storage).free < floor + headroom:
        raise SystemExit('Insufficient free space for build headroom and a 5 GiB reserve; no build started.')
    output = ROOT / 'outputs/runtime'
    output.mkdir(parents=True, exist_ok=True)
    command = ['docker', 'build', '--platform', 'linux/amd64', '--progress', 'plain',
               '--tag', 'pipknet-reproduction:20261114-v1', str(ROOT / 'reproducibility/runtime_recipe')]
    with (output / 'build.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if shutil.disk_usage(storage).free < floor:
                process.terminate()
                raise RuntimeError('Docker client stopped at the 5 GiB disk reserve; inspect Docker build status.')
            time.sleep(2)
    print('\n'.join((output / 'build.log').read_text(encoding='utf-8').splitlines()[-20:]))
    raise SystemExit(process.returncode)
if __name__ == '__main__':
    main()
