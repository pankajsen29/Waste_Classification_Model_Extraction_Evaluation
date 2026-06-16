#####################################################
# For each image: 
# - call surrogate API
# - save predictions in surrogate_results.jsonl
# - output: surrogate_results.jsonl
#
# - example json received:
# {
#   "image": "train/plastic/image_01.jpg",
#   "predicted_class": "Plastic",
#   "scores": 
#    {
#      "Cardboard": 0.9753, 
#      "Food Organics": 0.0003, 
#      "Glass": 0.0003, 
#      "Metal": 0.0015, 
#      "Miscellaneous Trash": 0.0021, 
#      "Paper": 0.0153, 
#      "Plastic": 0.0004, 
#      "Textile Trash": 0.003, 
#      "Vegetation": 0.0018
#   }
# }
# 
# - example json stored:
# {
#   "image": "train/plastic/image_01.jpg",
#   "predicted_class": "Plastic",
#   "scores": 
#    {
#      "Cardboard": 0.9753, 
#      "Food Organics": 0.0003, 
#      "Glass": 0.0003, 
#      "Metal": 0.0015, 
#      "Miscellaneous Trash": 0.0021, 
#      "Paper": 0.0153, 
#      "Plastic": 0.0004, 
#      "Textile Trash": 0.003, 
#      "Vegetation": 0.0018
#   }
#   "true_class": "plastic"
# }
#####################################################

import requests

# Configuration
API_URL = "http://127.0.0.1:8001/predict"
DATA_DIR = "data/TrashBox"

def get_surrogate_query_results(image_path):
    """
    Query surrogate API and store results
    """
    query_status = True
         
    relative_path = image_path.relative_to(DATA_DIR)
    #print(f"Processing: {relative_path}")
    true_class = relative_path.parts[1]

    mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png"
            }
    
    # "image_path" should be used for filesystem access, not the "relative_path"
    content_type = mime_types.get(image_path.suffix.lower(), "application/octet-stream")

    try:
        with open(image_path, "rb") as img_file:

            files = {
                "file": (
                    str(relative_path.as_posix()),
                    img_file,
                    content_type
                )
            }

            response = requests.post(
                API_URL,
                files=files,
                timeout=30
            )

        response.raise_for_status()
        prediction_result = response.json()
        prediction_result["true_class"] = true_class           
        query_status = True

    except Exception as e:
        query_status = False
        print(f"Failed Surrogate prediction for: {relative_path} -> {e}")
    
    return prediction_result, query_status
    