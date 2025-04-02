import ijson
import json
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

chunk_size = 10_000
input_file = 'data_nas_bench201.json'
output_prefix = 'data_bench201_part_'

chunk = []
file_index = 0
count = 0

with open(input_file, 'r') as f:
    # Parse each item in the top-level list
    for item in ijson.items(f, 'item'):
        chunk.append(item)
        count += 1

        if count % chunk_size == 0:
            output_file = f"{output_prefix}{file_index}.json"
            with open(output_file, 'w') as out_file:
                json.dump(chunk, out_file, indent=2, cls=DecimalEncoder)
            print(f"Saved {output_file} with {len(chunk)} entries")
            chunk = []
            file_index += 1

    # Save remaining items
    if chunk:
        output_file = f"{output_prefix}{file_index}.json"
        with open(output_file, 'w') as out_file:
            json.dump(chunk, out_file, indent=2, cls=DecimalEncoder)
        print(f"Saved {output_file} with {len(chunk)} entries")
