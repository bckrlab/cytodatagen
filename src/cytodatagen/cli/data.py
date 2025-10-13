#!/usr/bin/env python

import argparse
import dataclasses as dc
import json

from pathlib import Path


@dc.dataclass
class DataGeneratorConfig:
    pass


class DataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config

    def generate(self):
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=Path, help="path to the subjects.json file")
    parser.add_argument("--n-samples-per-subject", type=int, default=10)

    args = parser.parse_args()

    with open(args.subjects, "r") as subjects_json:
        subjects = json.load(subjects_json)


if __name__ == "__main__":
    main()
