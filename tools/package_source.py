"""Create a source-only archive using a positive file-type allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.py','.h','.hpp','.inl','.cpp','.cu','.cuh','.c','.cc','.in',
            '.md','.txt','.json','.toml','.yml','.yaml','.sh','.ps1'}
DOTFILES={'.gitignore','.gitattributes','.editorconfig'}
NAMES={'LICENSE','COPYING','NOTICE','MANIFEST.in'}
EXCLUDE={'.git','.idea','.vscode','.pytest_cache','__pycache__','.venv','venv',
         'runs','results','data','datasets','weights','checkpoints','outputs',
         'artifacts','native','build','dist','validation'}

def sources(root=ROOT):
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or p.is_symlink() or any(v in EXCLUDE or v.endswith('.egg-info') for v in rel.parts):continue
        if p.suffix.lower() in EXTENSIONS or p.name in DOTFILES or p.name in NAMES:
            yield p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new archive path')
    files=list(sources())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in files: archive.write(p,'AdaSplat/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip():raise RuntimeError('ZIP integrity check failed')
    print(json.dumps(dict(files=len(files),bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
