import google.generativeai as genai
import json, os, logging, time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GeminiTrainer:
    def __init__(self, output_dir: str = "training_data"):
        keys_str = os.getenv('GEMINI_API_KEYS')
        if not keys_str:
            raise ValueError("GEMINI_API_KEYS not set.")
        
        self.api_keys = [key.strip() for key in keys_str.split(',')]
        self.current_key_index = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized with {len(self.api_keys)} API keys.")

    def _get_next_model(self):
        key_to_use = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.logger.info(f"Using API Key #{self.current_key_index}...")
        genai.configure(api_key=key_to_use)
        return genai.GenerativeModel('gemini-1.5-flash')

    def create_training_data(self, pdf_paths: List[str]) -> str:
        from utils.layout_utils import LayoutExtractor
        extractor = LayoutExtractor()
        all_training_data = []
        
        for pdf_path in pdf_paths:
            self.logger.info(f"Processing {pdf_path} for training data...")
            
            # ✅ FIXED: Use existing method from LayoutExtractor
            blocks = extractor.extract_and_save_layout(pdf_path, output_dir="layout_data")
            
            if not blocks:
                continue

            for i in range(0, len(blocks), 200):
                batch = blocks[i:i + 200]
                labels = self._get_batch_labels(batch)
                for block, label in zip(batch, labels):
                    all_training_data.append({
                        **block,
                        'label': label,
                        'source_pdf': os.path.basename(pdf_path)
                    })

        training_file = os.path.join(self.output_dir, 'gemini_training_data.json')
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created {len(all_training_data)} training samples")
        return training_file

    def _get_batch_labels(self, blocks: List[Dict[str, Any]]) -> List[str]:
        prompt = self._create_batch_prompt(blocks)
        for attempt in range(len(self.api_keys)):
            try:
                model = self._get_next_model()
                response = model.generate_content(prompt)
                return self._parse_response(response.text, len(blocks))
            except Exception as e:
                if "429" in str(e):
                    self.logger.warning(f"Key #{self.current_key_index} rate-limited. Trying next...")
                else:
                    self.logger.error(f"Non-rate-limit error with key #{self.current_key_index}: {e}")
        self.logger.error(f"All API keys failed for batch. Returning NONEs.")
        return ['NONE'] * len(blocks)

    def _create_batch_prompt(self, blocks: List[Dict[str, Any]]) -> str:
        prompt = """You are a document hierarchy expert. Your task is to identify structural headings (TITLE, H1, H2, H3) based on a strict numbering and contextual system.

**DEFINITIVE HIERARCHY RULES:**

1.  **`TITLE`**: The main title on the cover page. Usually the largest font on page 1.
2.  **`H1`**: The highest-level sections. These are either:
    *   A single-digit numbered section (e.g., "1. Introduction").
    *   A major, un-numbered section title (e.g., "Abstract", "Table of Contents", "Acknowledgements").
3.  **`H2`**: Subsections. These are either:
    *   A two-digit numbered section (e.g., "2.1 Intended Audience").
    *   A non-bold subtitle that directly follows an H1.
4.  **`H3`**: Minor topic headers.
5.  **`NONE`**: Everything else. This is the default. Paragraphs, running headers, footers, and entries *within* a Table of Contents are always `NONE`.

---
**EXAMPLE:**
**Input:**
BLOCK 1: "2. Introduction" [Font: 14, bold]
BLOCK 2: "2.1 Intended Audience" [Font: 11, bold]
BLOCK 3: "2.1.1 ..."
BLOCK 4: "The certification is for professionals..." [Font: 11]
BLOCK 5: "• ..."
BLOCK 6: "- Additional detail"
BLOCK 7: "1) Step one"

**Output:**
BLOCK 1: H1
BLOCK 2: H2
BLOCK 3: H3
BLOCK 4: NONE
BLOCK 5: NONE
BLOCK 6: NONE
BLOCK 7: NONE
---

**TASK: Classify the following blocks using ONLY these definitive rules.**
**Input Blocks:**
"""
        for i, block in enumerate(blocks):
            style = [s for s, b in [("bold", block.get('is_bold')), ("italic", block.get('is_italic'))] if b]
            style_str = f", Style: {', '.join(style)}" if style else ""
            prompt += f"\nBLOCK {i+1}: \"{block.get('text', '')}\"\n"
            prompt += f"[Font size: {block.get('font_size', 0):.2f}{style_str}, Page: {block.get('page_number', 0)}]"
        prompt += "\n---\n**Output:**"
        return prompt

    def _parse_response(self, response_text: str, expected_count: int) -> List[str]:
        labels, label_map = [], {}
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('BLOCK ') and ':' in line:
                try:
                    parts = line.split(':', 1)
                    block_num = int(parts[0].replace('BLOCK', '').strip())
                    label = parts[1].strip().split(" ")[0]
                    if label in ['TITLE', 'H1', 'H2', 'H3', 'NONE']:
                        label_map[block_num] = label
                except (ValueError, IndexError):
                    continue
        for i in range(1, expected_count + 1):
            labels.append(label_map.get(i, 'NONE'))
        return labels
