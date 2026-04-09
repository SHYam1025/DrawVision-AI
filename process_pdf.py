import os
import cv2
import numpy as np
import pdf2image
import pandas as pd
import json
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

# Bypass the DecompressionBombWarning for massive engineering drawings
Image.MAX_IMAGE_PIXELS = None 

from google import genai
from google.genai import types

# --- SYSTEM CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

YOLO_MODEL_PATH = "yolo26m.pt"  
INFERENCE_IMGSZ = 1024       
CONF_THRESHOLD = 0.20        
CROP_PADDING = 30           

PDF_DPI_RESOLUTION = 300     
OUTPUT_CSV_NAME = "anchors_text02.csv"
OUTPUT_IMG_DIR = "annotated_images" # NEW: Folder to save drawn images
# -----------------------------

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Please check your .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)

def setup_gemini_prompt():
    return """
    Extract the piping specification text from this image crop.
    WARNING: There may be MULTIPLE different pipe specifications stacked in this single image crop.
    You MUST return a JSON ARRAY containing a separate object for EACH pipe specification found.
    Return ONLY a JSON array. If a value is missing or cannot be read, use an empty string "".

    [
      {
        "text": "The full combined raw text for this specific pipe",
        "project_name": "extracted project or area code (e.g., 335)",
        "diameter": "extracted diameter (e.g., 2\\", 1.5\\", 0.75\\")",
        "service_code": "extracted service code (e.g., PR, WF, SL)",
        "line_number": "extracted line sequence number (e.g., 195, 004, 197)",
        "piping_class": "extracted piping class (e.g., B02F20B, A03F00D, A01F26A-HC)",
        "bop_elevation": "extracted BOP number if present below it (e.g., 115500, 116505, 117600)"
      }
    ]
    """

def perform_robust_crop(original_image, xyxy_coords, padding):
    x1, y1, x2, y2 = xyxy_coords
    width, height = original_image.size

    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(width, int(x2) + padding)
    y2 = min(height, int(y2) + padding)

    return original_image.crop((x1, y1, x2, y2))

def gemini_ocr_single_crop(crop_pil, prompt_instructions):
    img_byte_arr = BytesIO()
    crop_pil.save(img_byte_arr, format='JPEG') 
    img_bytes = img_byte_arr.getvalue()
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                prompt_instructions,
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "[]" 

def run_system_on_pdf(pdf_path_input):
    print(f"\n--- Launching Anchors Processing System for {pdf_path_input} ---")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ Error: YOLO model {YOLO_MODEL_PATH} not found in this directory.")
        return

    # NEW: Create the output folder for images if it doesn't exist
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    model_yolo = YOLO(YOLO_MODEL_PATH)
    
    print("Step 1: Converting PDF pages to high-resolution images...")
    try:
        pages_pil = pdf2image.convert_from_path(pdf_path_input, dpi=PDF_DPI_RESOLUTION)
    except Exception as e:
        print(f"❌ Error during PDF conversion. Error: {e}")
        return

    system_prompt = setup_gemini_prompt()
    all_data_for_csv = []
    
    for page_num, original_page_pil in enumerate(pages_pil):
        current_page_no = page_num + 1
        print(f"\n- Analysing Page {current_page_no} -")
        
        img_cv2_orig = cv2.cvtColor(np.array(original_page_pil), cv2.COLOR_RGB2BGR)

        print(f"  Step 2: Running YOLO (Processing Size: {INFERENCE_IMGSZ})...")
        results = model_yolo.predict(
            source=img_cv2_orig, 
            imgsz=INFERENCE_IMGSZ, 
            conf=CONF_THRESHOLD, 
            verbose=False
        )
        
        detections = results[0] 
        count = len(detections.boxes)
        print(f"  ✅ Detected {count} bounding boxes.")

        if count == 0: continue

        boxes_xyxy = detections.boxes.xyxy.cpu().numpy() 
        confs = detections.boxes.conf.cpu().numpy()

        # --- NEW: Draw bounding boxes and save the image ---
        annotated_img = img_cv2_orig.copy()
        for i in range(count):
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            # Draw a Green box with a thickness of 5 (since the image is very high-res)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # Save the drawn image to the folder
        img_filename = f"{os.path.splitext(os.path.basename(pdf_path_input))[0]}_page_{current_page_no}_annotated.jpg"
        img_save_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
        cv2.imwrite(img_save_path, annotated_img)
        print(f"  🖼️ Saved visual YOLO output to: {img_save_path}")
        # ---------------------------------------------------

        print(f"  Step 3: Cropping high-res original and running Multi-Extraction JSON...")

        for i in range(count):
            x1, y1, x2, y2 = boxes_xyxy[i]
            crop_pil = perform_robust_crop(original_page_pil, boxes_xyxy[i], CROP_PADDING)

            if crop_pil:
                raw_json_string = gemini_ocr_single_crop(crop_pil, system_prompt)
                
                try:
                    parsed_data_list = json.loads(raw_json_string)
                    if isinstance(parsed_data_list, dict):
                        parsed_data_list = [parsed_data_list]
                except json.JSONDecodeError:
                    print(f"    ⚠️ Warning: Failed to parse JSON for Box {i+1}.")
                    parsed_data_list = []

                for parsed_data in parsed_data_list:
                    all_data_for_csv.append({
                        'image_name': f"{os.path.basename(pdf_path_input)}_page_{current_page_no}",
                        'class_id': 0, 
                        'xmin': int(x1),
                        'ymin': int(y1),
                        'xmax': int(x2),
                        'ymax': int(y2),
                        'confidence': round(float(confs[i]), 4),
                        'text': parsed_data.get("text", ""),
                        'project_name': parsed_data.get("project_name", ""),
                        'line_number': parsed_data.get("line_number", ""),
                        'diameter': parsed_data.get("diameter", ""),
                        'service_code': parsed_data.get("service_code", ""),
                        'piping_class': parsed_data.get("piping_class", ""),
                        'bop_elevation': parsed_data.get("bop_elevation", "")
                    })
                    print(f"    - Extracted Pipe: {parsed_data.get('diameter')} {parsed_data.get('service_code')} line")

    # --- CSV Output Generation ---
    if all_data_for_csv:
        df_results = pd.DataFrame(all_data_for_csv)
        
        columns_order = [
            'image_name', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax', 
            'confidence', 'text', 'project_name', 'line_number', 
            'diameter', 'service_code', 'piping_class', 'bop_elevation'
        ]
        df_results = df_results[columns_order]
        
        df_results['text'] = df_results['text'].replace(r'\n', ' ', regex=True)
        
        df_results['box_signature'] = df_results['xmin'].astype(str) + "_" + df_results['ymin'].astype(str)
        
        df_results['zone_x'] = (df_results['xmin'] // 100) * 100
        df_results['zone_y'] = (df_results['ymin'] // 100) * 100
        
        valid_boxes = df_results.drop_duplicates(subset=['image_name', 'zone_x', 'zone_y'])['box_signature'].tolist()
        
        df_results = df_results[df_results['box_signature'].isin(valid_boxes)]
        df_results = df_results.drop(columns=['box_signature', 'zone_x', 'zone_y'])
        
        df_results.to_csv(OUTPUT_CSV_NAME, index=False, header=True, encoding='utf-8')
        
        print(f"\n✅ Success! Saved perfectly structured data to {OUTPUT_CSV_NAME}")
        print(f"📊 Total unique physical pipes extracted: {len(df_results)}") 
    else:
        print("\n❌ No data extracted to save.")
    
if __name__ == "__main__":
    example_pdf_file = "my_engineering_drawing01.pdf" 
    if os.path.exists(example_pdf_file):
        run_system_on_pdf(example_pdf_file)
    else:
        print(f"Error: Please place '{example_pdf_file}' in this folder.")