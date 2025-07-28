import fitz
import logging
import os
import json
import statistics
from typing import List, Dict, Any


class LayoutExtractor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def extract_and_save_layout(self, doc_path: str, output_dir: str) -> List[Dict[str, Any]]:
        doc_name = os.path.basename(doc_path)
        base_name = os.path.splitext(doc_name)[0]
        print(f"  - Processing document: {doc_name}")
        layout_data = []

        try:
            doc = fitz.open(doc_path)

            # Step 1: Compute base font size using median
            all_font_sizes = []
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_font_sizes.append(span["size"])
            base_font_size = statistics.median(all_font_sizes) if all_font_sizes else 10

            for page_num, page in enumerate(doc):
                page_width = page.rect.width
                page_height = page.rect.height
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    block_lines = []
                    max_font_size = 0
                    bold_flags = []
                    italic_flags = []
                    all_spans = []

                    for line in block["lines"]:
                        line_text = "".join([span.get("text", "").strip() for span in line["spans"]])
                        if not line_text.strip():
                            continue
                        block_lines.append(line_text)

                        for span in line["spans"]:
                            max_font_size = max(max_font_size, span["size"])
                            bold_flags.append("bold" in span.get("font", "").lower())
                            italic_flags.append("italic" in span.get("font", "").lower())
                            all_spans.append(span)

                    if not block_lines:
                        continue

                    block_text = " ".join(block_lines).strip()
                    x0, y0, x1, y1 = block["bbox"]

                    block_data = {
                        'text': block_text,
                        'bbox': {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1},
                        'font_size': round(max_font_size, 2),
                        'is_bold': any(bold_flags),
                        'is_italic': any(italic_flags),
                        'page_number': page_num + 1,
                        'line_position': y0,
                        'width': x1 - x0,
                        'height': y1 - y0,
                        'relative_x': x0 / page_width if page_width > 0 else 0,
                        'relative_y': y0 / page_height if page_height > 0 else 0,
                        'page_width': page_width,
                        'page_height': page_height,
                        'source': 'digital'
                    }

                    layout_data.append(block_data)

            # Save layout data
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2, ensure_ascii=False)
            print(f"  - Layout data saved to: {output_path}")

        except Exception as e:
            print(f"[ERROR] Failed to extract layout from {doc_name}: {e}")
            return []

        return layout_data
