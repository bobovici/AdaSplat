"""Install pinned dependencies and build the CUDA renderer in the active environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-torch',action='store_true',help='Keep an already installed compatible PyTorch')
    parser.add_argument('--no-build',action='store_true',help='Install dependencies only')
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if not (3,8)<=sys.version_info[:2]<(3,12):
        parser.error('Use Python 3.8--3.11 (3.10 recommended on a fresh server)')
    if args.jobs<1: parser.error('jobs must be positive')
    def run(*command): subprocess.run([sys.executable,*command],cwd=str(ROOT),check=True)
    if not args.skip_torch:
        run('-m','pip','install','torch==2.4.1+cu118','--index-url','https://download.pytorch.org/whl/cu118')
    run('-m','pip','install','-r','requirements.txt','-r','requirements-codec.txt','-r','requirements-build.txt')
    run('-m','pip','install','-e','.','--no-deps','--no-build-isolation')
    if not args.no_build:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if not CUDA_HOME: parser.error('CUDA toolkit not found; install CUDA 11.8 or set CUDA_HOME')
        nvcc=Path(CUDA_HOME)/'bin'/('nvcc.exe' if os.name=='nt' else 'nvcc')
        version=subprocess.check_output([str(nvcc),'--version'],text=True)
        if 'release '+str(torch.version.cuda) not in version:
            parser.error('CUDA toolkit and PyTorch CUDA versions differ. Select a matching toolkit/PyTorch pair before compiling')
        run('tools/build_renderer.py','--jobs',str(args.jobs))
    run('tools/check_env.py')

if __name__=='__main__':main()
