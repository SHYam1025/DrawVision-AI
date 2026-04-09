import os
import cv2
import numpy as np
import pdf2image
import pandas as pd
import json
import asyncio
import shutil
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
OUTPUT_IMG_DIR = "annotated_images"
UPLOAD_DIR = "uploads"

# Ensure directories exist
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Please check your .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)
# Load YOLO model once
model_yolo = YOLO(YOLO_MODEL_PATH)

app = FastAPI(title="DrawVision AI API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for frontend and annotated images
app.mount("/images", StaticFiles(directory=OUTPUT_IMG_DIR), name="images")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

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

async def gemini_ocr_single_crop_async(crop_pil, prompt_instructions):
    img_byte_arr = BytesIO()
    crop_pil.save(img_byte_arr, format='JPEG') 
    img_bytes = img_byte_arr.getvalue()
    
    try:
        # Note: genai client might not be fully async, but we treat it as such or wrap it
        response = await asyncio.to_thread(
            client.models.generate_content,
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

# Store results in memory for now (simplicity)
# In a real app, use a DB
results_cache = {}

@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Wait for the filename to process
        data = await websocket.receive_json()
        filename = data.get("filename")
        if not filename:
            await websocket.send_json({"error": "No filename provided"})
            return

        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            await websocket.send_json({"error": "File not found"})
            return

        await websocket.send_json({"status": "starting", "message": f"Processing {filename}..."})
        
        # Start processing
        pages_pil = await asyncio.to_thread(pdf2image.convert_from_path, filepath, dpi=PDF_DPI_RESOLUTION)
        await websocket.send_json({"status": "info", "message": f"Converted PDF to {len(pages_pil)} pages."})

        system_prompt = setup_gemini_prompt()
        all_data = []
        
        for page_num, original_page_pil in enumerate(pages_pil):
            current_page_no = page_num + 1
            await websocket.send_json({"status": "progress", "page": current_page_no, "total_pages": len(pages_pil), "message": f"Analyzing Page {current_page_no}..."})
            
            img_cv2_orig = cv2.cvtColor(np.array(original_page_pil), cv2.COLOR_RGB2BGR)
            
            # Run YOLO
            results = await asyncio.to_thread(
                model_yolo.predict,
                source=img_cv2_orig, 
                imgsz=INFERENCE_IMGSZ, 
                conf=CONF_THRESHOLD, 
                verbose=False
            )
            
            detections = results[0] 
            count = len(detections.boxes)
            await websocket.send_json({"status": "info", "message": f"Page {current_page_no}: Detected {count} bounding boxes."})

            if count == 0: continue

            boxes_xyxy = detections.boxes.xyxy.cpu().numpy() 
            confs = detections.boxes.conf.cpu().numpy()

            # Draw and save annotated image
            annotated_img = img_cv2_orig.copy()
            for i in range(count):
                bx1, by1, bx2, by2 = map(int, boxes_xyxy[i])
                cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 255, 0), 5)
            
            img_filename = f"{os.path.splitext(filename)[0]}_page_{current_page_no}_annotated.jpg"
            img_save_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
            cv2.imwrite(img_save_path, annotated_img)
            
            # Send image update to frontend
            await websocket.send_json({
                "status": "detection_image", 
                "page": current_page_no,
                "image_url": f"/images/{img_filename}",
                "count": count
            })

            # Extraction
            for i in range(count):
                x1, y1, x2, y2 = boxes_xyxy[i]
                crop_pil = perform_robust_crop(original_page_pil, boxes_xyxy[i], CROP_PADDING)

                if crop_pil:
                    raw_json_string = await gemini_ocr_single_crop_async(crop_pil, system_prompt)
                    
                    try:
                        parsed_data_list = json.loads(raw_json_string)
                        if isinstance(parsed_data_list, dict):
                            parsed_data_list = [parsed_data_list]
                    except json.JSONDecodeError:
                        parsed_data_list = []

                    for parsed_data in parsed_data_list:
                        item = {
                            'id': len(all_data) + 1,
                            'image_name': f"{filename}_page_{current_page_no}",
                            'confidence': round(float(confs[i]), 4),
                            'text': parsed_data.get("text", ""),
                            'project_name': parsed_data.get("project_name", ""),
                            'line_number': parsed_data.get("line_number", ""),
                            'diameter': parsed_data.get("diameter", ""),
                            'service_code': parsed_data.get("service_code", ""),
                            'piping_class': parsed_data.get("piping_class", ""),
                            'bop_elevation': parsed_data.get("bop_elevation", "")
                        }
                        all_data.append(item)
                        # Send live extraction update
                        await websocket.send_json({"status": "extraction", "data": item})

        # Save results to cache
        results_cache[filename] = all_data
        
        # Post-processing (Deduplication as in original script)
        if all_data:
            df = pd.DataFrame(all_data)
            # Simplified deduplication for the POC
            # (In production, you'd want the exact same logic as process_pdf.py)
            await websocket.send_json({"status": "complete", "message": "Processing complete!", "total_results": len(all_data)})
        else:
            await websocket.send_json({"status": "complete", "message": "No data extracted.", "total_results": 0})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in processing: {e}")
        await websocket.send_json({"status": "error", "message": str(e)})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/results/{filename}")
async def get_results(filename: str):
    if filename not in results_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    return results_cache[filename]

@app.get("/download/{format}/{filename}")
async def download_results(format: str, filename: str):
    if filename not in results_cache:
        # Try to load from existing CSV if cached is empty
        # For simplicity, we assume it's in results_cache
        raise HTTPException(status_code=404, detail="Results not found")
    
    data = results_cache[filename]
    df = pd.DataFrame(data)
    
    export_filename = f"results_{os.path.splitext(filename)[0]}"
    
    if format == "csv":
        export_path = f"{export_filename}.csv"
        df.to_csv(export_path, index=False)
        return FileResponse(export_path, filename=f"{export_filename}.csv")
    elif format == "excel":
        export_path = f"{export_filename}.xlsx"
        df.to_excel(export_path, index=False)
        return FileResponse(export_path, filename=f"{export_filename}.xlsx")
    else:
        raise HTTPException(status_code=400, detail="Invalid format")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
