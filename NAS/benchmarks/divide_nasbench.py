import ijson
import json
from decimal import Decimal
import os
from src.utils.main_utils import download_data


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


if __name__ == "__main__":
    """
    Split a large JSON file into smaller chunks while handling Decimal numbers.

    This script reads a large JSON file containing a list of items and splits it into smaller JSON files
    with a specified number of items per file. It handles Decimal numbers by converting them to float
    during the JSON serialization process.

    Classes:
        DecimalEncoder: A custom JSON encoder that converts Decimal objects to float.

    Configuration:
        - chunk_size: Number of items per output file (default: 10,000)
        - output_dir: Directory path for output files
        - input_file: Path to the input JSON file
        - output_prefix: Prefix for output file names

    Output:
        Creates multiple JSON files named 'data_part_X.json' where X is a sequential number,
        each containing up to chunk_size items from the input file.

    Example:
        Input file 'data.json' with 25,000 items and chunk_size=10,000 will create:
        - data_part_0.json (10,000 items)
        - data_part_1.json (10,000 items)
        - data_part_2.json (5,000 items)

    Note:
        The script uses ijson for memory-efficient parsing of large JSON files,
        making it suitable for processing large datasets that don't fit in memory.
    """
    output_dir = "NAS/benchmarks/nasbench101/dataset"
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = 10_000
    input_file = os.path.join(output_dir, "data.json")
    if not os.path.isfile(input_file):
        download_data(url="17jCPgF8TnbAAISt1UB3Hmx8kJu1UBs6l",
                      output = 'NAS/benchmarks/nasbench101/dataset/data.json')
    output_prefix = os.path.join(output_dir, "data_part_")

    chunk = []
    file_index = 0
    count = 0

    with open(input_file, "r") as f:
        # Parse each item in the top-level list
        for item in ijson.items(f, "item"):
            chunk.append(item)
            count += 1

            if count % chunk_size == 0:
                output_file = f"{output_prefix}{file_index}.json"
                with open(output_file, "w") as out_file:
                    json.dump(chunk, out_file, indent=2, cls=DecimalEncoder)
                print(f"Saved {output_file} with {len(chunk)} entries")
                chunk = []
                file_index += 1

        # Save remaining items
        if chunk:
            output_file = f"{output_prefix}{file_index}.json"
            with open(output_file, "w") as out_file:
                json.dump(chunk, out_file, indent=2, cls=DecimalEncoder)
            print(f"Saved {output_file} with {len(chunk)} entries")
