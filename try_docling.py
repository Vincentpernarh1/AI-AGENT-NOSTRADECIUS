# import os
# import json
# from pathlib import Path
# from docling.document_converter import DocumentConverter
# from docling.datamodel.document import PictureItem

# # Optional: disable HF symlink warnings
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# # Input and output
# source = "data_files/CID.pptx"
# output_dir = Path("docling_output")
# images_dir = output_dir / "images"
# output_dir.mkdir(exist_ok=True)
# images_dir.mkdir(exist_ok=True)

# # Convert document
# converter = DocumentConverter()
# result = converter.convert(source)
# doc = result.document

# # JSON structure
# ai_output = {"pages": []}
# page_index = 1
# page_data = {"number": page_index, "content": []}
# img_counter = 0

# # Traverse items in reading order
# for item, _ in doc.iterate_items():
#     # Determine current page
#     current_page = getattr(item, "page_no", page_index)
    
#     # Start new page if needed
#     if current_page != page_index:
#         ai_output["pages"].append(page_data)
#         page_index = current_page
#         page_data = {"number": page_index, "content": []}

#     # Text block
#     if hasattr(item, "text") and item.text and item.text.strip():
#         page_data["content"].append({
#             "type": "text",
#             "text": item.text.strip()
#         })

#     # Image block
#     if isinstance(item, PictureItem):
#         img = item.get_image(doc)
#         if img:
#             img_filename = f"page{page_index}_img{img_counter}.png"
#             img_path = images_dir / img_filename
#             img.save(img_path)
#             page_data["content"].append({
#                 "type": "image",
#                 "path": f"images/{img_filename}",
#                 "description": f"Image on page {page_index}"
#             })
#             img_counter += 1

# # Append last page
# if page_data:
#     ai_output["pages"].append(page_data)

# # Save AI-friendly JSON
# json_file = output_dir / "document.json"
# with open(json_file, "w", encoding="utf-8") as f:
#     json.dump(ai_output, f, indent=2, ensure_ascii=False)

# print(f"✅ Done!\nAI JSON: {json_file}\nImages folder: {images_dir}")




import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.document import PictureItem

# Optional: disable HF symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Input and output
source = "data_files/CID.pptx"
output_dir = Path("docling_output")
images_dir = output_dir / "images"
output_dir.mkdir(exist_ok=True)
images_dir.mkdir(exist_ok=True)

# Convert document
converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

# Keywords for heuristic labeling
FIELD_KEYWORDS = {
    "Supplier": ["supplier", "fornecedor"],
    "Investment": ["investment", "investimento", "r$"],
    "Proposal Capacity": ["jph", "jpd", "jpw", "jpy", "capacity"],
    "SAP/Cofor": ["sap", "cofor"],
    "State/Country": ["state", "country", "cidade", "estado"]
}

# Helper to assign labels based on text
def detect_label(text):
    t = text.lower()
    for key, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return key
    return None

# JSON structure
ai_output = {"pages": []}
page_index = 1
page_data = {"number": page_index, "content": []}
img_counter = 0

# Traverse items in reading order
for item, _ in doc.iterate_items():
    current_page = getattr(item, "page_no", page_index)
    
    if current_page != page_index:
        ai_output["pages"].append(page_data)
        page_index = current_page
        page_data = {"number": page_index, "content": []}

    # Text block
    if hasattr(item, "text") and item.text and item.text.strip():
        label = detect_label(item.text)
        text_obj = {
            "type": "text",
            "text": item.text.strip(),
            "bbox": getattr(item, "bbox", None)  # preserve bounding box
        }
        if label:
            text_obj["field"] = label
        page_data["content"].append(text_obj)

    # Image block
    if isinstance(item, PictureItem):
        img = item.get_image(doc)
        if img:
            img_filename = f"page{page_index}_img{img_counter}.png"
            img_path = images_dir / img_filename
            img.save(img_path)
            page_data["content"].append({
                "type": "image",
                "path": f"images/{img_filename}",
                "description": f"Image on page {page_index}"
            })
            img_counter += 1

# Append last page
if page_data:
    ai_output["pages"].append(page_data)

# Save AI-friendly JSON
json_file = output_dir / "document.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(ai_output, f, indent=2, ensure_ascii=False)

print(f"✅ Done!\nAI JSON: {json_file}\nImages folder: {images_dir}")
