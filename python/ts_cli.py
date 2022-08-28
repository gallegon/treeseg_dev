"""CLI bindings for running the tree segmentation algorithm.

To use from the command line, run: ``ts_cli.py context_file``
"""

import json
import sys

from treesegmentation.ts_api import default_pipeline
from integration_with_c import c_pipeline


def load_context_data(file_path):
    """Helper function to load a json dictionary from a file.

    :param file_path: Input file path of the JSON file to load.

    :return: On success, returns a Python dictionary containing the parsed file.
    """
    with open(file_path) as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("Context file must be a JSON dictionary.")
    return data


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        context = load_context_data(args[0])
        result = c_pipeline(context)
        elapsed_time = result["elapsed_time"]
        print(f"Pipeline completed {len(default_pipeline.handlers)} stages in {elapsed_time} seconds.")
    else:
        print("ts_cli expects a path to a context file path as its only argument.")
        print("Usage:")
        print("    ts_cli.py context_file")


if __name__ == "__main__":
    main()
