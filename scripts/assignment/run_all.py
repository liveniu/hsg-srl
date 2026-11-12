#!/usr/bin/env python3
"""Run the clock check and the Table S3 closure. Writes only results/."""

from clock_check import main as clock_main
from table_s3_closure import main as table_main


def main() -> None:
    clock_main()
    table_main()


if __name__ == "__main__":
    main()
