#!/usr/bin/env python3

import logging
from text_sample import create_ghostbuster_datasets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    create_ghostbuster_datasets()
    print("Ghostbuster datasets successfully created!")
