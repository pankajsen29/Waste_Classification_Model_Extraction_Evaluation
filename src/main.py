################ Step 1: iterates all the images from DATA_DIR and queries target and surrogate APIs for predictions ##################
import json
from pathlib import Path
from src.query.query_target_api import get_target_query_results
from src.query.query_surrogate_api import get_surrogate_query_results

DATA_DIR = "data/TrashBox"
TARGET_OUTPUT_FILE = "data/Query_Results/target_results.jsonl"
SURROGATE_OUTPUT_FILE = "data/Query_Results/surrogate_results.jsonl"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

with open(Path(TARGET_OUTPUT_FILE), "w", encoding="utf-8") as target_out_file:
    with open(Path(SURROGATE_OUTPUT_FILE), "w", encoding="utf-8") as surrogate_out_file:
         
        for image_path in Path(DATA_DIR).rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                        continue
            
            # query target API
            target_prediction_result, target_query_status = get_target_query_results(image_path)
            if target_query_status:
                target_out_file.write(json.dumps(target_prediction_result) + "\n")

                # query surrogate API
                surrogate_prediction_result, surrogate_query_status = get_surrogate_query_results(image_path)
                if surrogate_query_status:
                    surrogate_out_file.write(json.dumps(surrogate_prediction_result) + "\n")
            else:
                continue

print(f"\nTarget query results is stored in {Path(TARGET_OUTPUT_FILE)}")
print(f"\nSurrogate query results is stored in {Path(SURROGATE_OUTPUT_FILE)}")           
