import base64
import requests
import os
import json
import csv
import re

BASE_DIR = "D:/Work/output"
OLLAMA_URL = "http://184.105.6.66:11434/api/generate"
MODEL_NAME = "llava:34b"
OUTPUT_CSV = "D:/Work/output/results.csv"

PROMPT_TEXT = '''
Prompt:
You are a classification assistant. Please complete the following tasks. Each task is output in a separate line and starts with the task number "1. ":
1. First, describe the objects shown in the rendered image in one sentence. The constraints on this description are: you do not need to describe the background. Please accurately describe all the objects in the image. Avoid imagination and fiction. Do not include any additional comments or unnecessary details. Do not include meaningless content such as "This image is..." or "A 3D model of..."; please directly output the sentence in plain text, without any extra explanatory text, markdown, bullets or special formatting.
2. Then, according to the sentence you just gave, remove all descriptive and descriptive text, and only keep the objects in the image. The constraints on this text are: it does not need to be a complete sentence, no adjectives are required, and only the most concise text is needed to provide information about the objects in the image. If there is more than one object in the image, list them all and separate them with commas; also, please directly output the content in plain text, without any extra explanatory text, markdown, bullets or special formatting.
3. Finally, please merge the above two lines into the format I need. For the results of Task 1 and Task 2, please output them in the format of ["Results of Task 1", "Results of Task 2"].
Here is an example of the output format:
1. Golden-armored warrior with angelic features, including sleek helmet, robust torso armor, large shoulder pads, segmented arm and leg armor, elaborate engravings, expansive detailed wings, and an ornate bow and arrow in various dynamic combat stances. 
2. Armored warrior. 
3. ["Golden-armored warrior with angelic features, including sleek helmet, robust torso armor, large shoulder pads, segmented arm and leg armor, elaborate engravings, expansive detailed wings, and an ornate bow and arrow in various dynamic combat stances.", "Armored warrior."]
Return the response in this structured format.
'''

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except Exception as e:
        print(f"[Error] Encoding image {image_path} failed: {e}")
        return None

def send_request_to_ollama(encoded_images, temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEXT,
        "images": encoded_images,
        "temperature": temperature
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True)
        resp.raise_for_status()

        final_response = ""
        for line in resp.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                # print(f"Raw response line: {decoded_line}")
                json_data = json.loads(decoded_line)
                if "response" in json_data:
                    final_response += json_data["response"]
                if json_data.get("done", False):
                    break

        print(f"Full response: '{final_response}'")
        
        # 使用正则表达式提取第 3 部分的 JSON 列表
        json_match = re.search(r'3\.\s*(\[.*?\])', final_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"Extracted JSON: '{json_str}'")
            return json.loads(json_str)
        else:
            print("[Error] Could not extract JSON list from response")
            return None
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[Error] Request or JSON parsing failed: {e}")
        return None

def process_subfolder(subfolder_path):
    image_paths = [
        os.path.join(subfolder_path, "000000.png"),
        os.path.join(subfolder_path, "000001.png"),
        os.path.join(subfolder_path, "000002.png"),
        os.path.join(subfolder_path, "000003.png")
    ]
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[Error] Image file not found: {image_path}")
            return None

    encoded_images = []
    for image_path in image_paths:
        encoded_image = encode_image_to_base64(image_path)
        if encoded_image is None:
            print(f"[Error] Failed to encode image: {image_path}")
            return None
        encoded_images.append(encoded_image)
    return send_request_to_ollama(encoded_images)

def main():
    subfolders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    if not subfolders:
        print(f"[Error] No subfolders found in {BASE_DIR}")
        return

    results = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(BASE_DIR, subfolder)
        print(f"\nProcessing subfolder: {subfolder}")
        response = process_subfolder(subfolder_path)
        if response is not None:
            print(f"Result for {subfolder}: {response}")
            results.append({"folder_name": subfolder, "result": response})
        else:
            print(f"Failed to process {subfolder}")
            results.append({"folder_name": subfolder, "result": None})

    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["folder_name", "description", "components"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            folder_name = result["folder_name"]
            response = result["result"]
            if response and isinstance(response, list) and len(response) == 2:
                description, components = response
                writer.writerow({
                    "folder_name": folder_name,
                    "description": description,
                    "components": components
                })
            else:
                writer.writerow({
                    "folder_name": folder_name,
                    "description": "N/A",
                    "components": "N/A"
                })
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()