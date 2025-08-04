import json
import os
from glob import glob

from tqdm import tqdm


def build_dataset_json(metadata_dir, output_path, sharegpt4v_default=""):
    """
    metadata_dir: Directory with one JSON file per image (including width, height, caption, key, etc.)
    output_path: Path to write the merged JSON list.
    sharegpt4v_default: Default GPT-4V caption if not available (empty by default).
    """
    all_entries = []
    json_files = sorted(glob(os.path.join(metadata_dir, "*.json")))

    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            data = json.load(f)

        key = data.get("key")
        width = data.get("width")
        height = data.get("height")
        caption = data.get("caption", "").strip()

        # Validation
        if not key or not width or not height or not caption:
            print(f"Skipping {json_file} due to missing fields.")
            continue

        ratio = round(width / height, 3)
        entry = {
            "height": height,
            "width": width,
            "ratio": ratio,
            "path": f"{key}.webp",
            "prompt": caption,
            "sharegpt4v": sharegpt4v_default.strip()
        }
        all_entries.append(entry)
    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"✅ Saved {len(all_entries)} entries to {output_path}")

# Example usage:
build_dataset_json(
    metadata_dir="/export/data/vislearn/rother_subgroup/dzavadsk/datasets/laion/subset_250/metadata",     # <- replace with your folder path
    output_path="/export/data/vislearn/rother_subgroup/sheid/pixart/laion/data_info.json",
    sharegpt4v_default=""
)
