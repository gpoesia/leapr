#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import platform
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def check_download(dest_path, url):
    """
    Check if the file at dest_path exists; if not, download it from url using wget.
    """
    if os.path.exists(dest_path):
        print(f"{dest_path} exists, skipping download.")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} to {dest_path} with curl...")
    try:
        subprocess.run(["curl", "-L", url, "-o", dest_path], check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e}")
        sys.exit(1)


def decompress(path, program: list[str]):
    """
    Check if `program` is available then run `program path` to decompress.
    """

    # Strip last extension from path and check if decompressed file already exists.
    decompressed_path = os.path.splitext(path)[0]
    if os.path.exists(decompressed_path):
        print(f"{decompressed_path} exists, skipping decompression.")
        return

    try:
        result = subprocess.run([program[0], "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, OSError):
        print(f"Error: '{program}' is not installed or not found in PATH.")
        sys.exit(1)

    print(f"Decompressing {path}...")
    try:
        subprocess.run([*program, path], check=True)
        print("Decompression complete.")
    except subprocess.CalledProcessError as e:
        print(f"Decompression failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        'groups', nargs='+',
        help="Groups of files to process (e.g., lichess-2014-01, lichess-2025-08, all)"
    )
    args = parser.parse_args()

    groups = {
        'lichess-pgn-2014-01': {
            'url': 'https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst',
            'dest': 'data/lichess-2014-01.pgn.zst',
            'decompress_prog': ['unzstd'],
        },
        'lichess-pgn-2025-08': {
            'url': 'https://database.lichess.org/standard/lichess_db_standard_rated_2025-08.pgn.zst',
            'dest': 'data/lichess-2025-08.pgn.zst',
            'decompress_prog': ['unzstd'],
        },
        'lichess-eval': {
            'url': 'https://database.lichess.org/lichess_db_eval.jsonl.zst',
            'dest': 'data/lichess-eval.jsonl.zst',
            'decompress_prog': ['unzstd'],
        },
        'stockfish': {
            'url': None,  # Set below, platform-specific.
            'dest': 'data/stockfish.tar',
            'decompress_prog': ['tar', 'xvf']
        }
    }

    # Set platform-specific downloads.
    if platform.system() == 'Linux':
        groups['stockfish']['url'] = 'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar'
    elif platform.system() == 'Darwin':
        groups['stockfish']['url'] = 'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-macos-m1-apple-silicon.tar'
    elif platform.system() == 'Windows':
        groups['stockfish']['url'] = 'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip'
    else:
        raise NotImplementedError

    to_run = set()
    if 'all' in args.groups:
        to_run = set(groups.keys())
    else:
        for g in args.groups:
            if g in groups:
                to_run.add(g)
            else:
                print(f"Warning: group '{g}' not recognized, skipping.")

    if not to_run:
        print("No valid groups specified.")
        sys.exit(1)

    for g in sorted(to_run):
        info = groups[g]
        check_download(info['dest'], info['url'])

        if info.get('decompress_prog'):
            decompress(info['dest'], info['decompress_prog'])

if __name__ == '__main__':
    main()
