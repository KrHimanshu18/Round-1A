import json, os, logging, re
from typing import List, Dict, Any
from collections import defaultdict

class PostProcessor:
    def __init__(self, output_dir: str = "output_json"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_predictions(self, blocks: List[Dict[str, Any]], predictions: List[str], pdf_name: str):
        try:
            cleaned_predictions = []
            for b, label in zip(blocks, predictions):
                text = b.get('text', '').strip()
                width = b['bbox']['x1'] - b['bbox']['x0']
                is_short_token = re.match(r'^(\d+[\.\)]?|[-•\u2022\u25AA\u25CF\u2023])$', text) is not None
                if is_short_token and width < 50:
                    cleaned_predictions.append('NONE')
                    continue
                cleaned_predictions.append(label)

            labeled_blocks = [b | {'label': p} for b, p in zip(blocks, cleaned_predictions) if p != 'NONE']

            
            toc_page_numbers = set()
            for block in labeled_blocks:
                if 'table of contents' in block.get('text', '').lower() and block.get('label') in ['H1', 'TITLE']:
                    toc_page_numbers.add(block['page_number'])
            if toc_page_numbers: self.logger.info(f"Identified TOC on page(s): {toc_page_numbers}")

            headings = self._extract_headings(labeled_blocks, toc_page_numbers)
            headings = self._correct_heading_levels(headings) # Apply hierarchy correction
            clean_headings = self._deduplicate_headings(headings) if headings else []
            
            final_data = {
                'pdf_name': pdf_name,
                'title': self._extract_title(clean_headings),
                'headings': self._structure_headings(clean_headings)
            }
            
            output_path = os.path.join(self.output_dir, f"{pdf_name}.json")
            with open(output_path, 'w', encoding='utf-8') as f: json.dump(final_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Processed {len(final_data['headings'])} final headings for {pdf_name}")
        except Exception as e:
            self.logger.error(f"Error processing predictions for {pdf_name}: {e}")

    def _correct_heading_levels(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Overrides AI predictions based on a strict, hard-coded numbering system."""
        corrected_headings = []
        for h in headings:
            text, original_level = h['text'], h['level']
            corrected_level = original_level
            if re.match(r'^\d+\.\d+\.\d+', text): corrected_level = 'H3'
            elif re.match(r'^\d+\.\d+', text): corrected_level = 'H2'
            elif re.match(r'^\d+\.', text): corrected_level = 'H1'
            if corrected_level != original_level and original_level != 'TITLE':
                self.logger.warning(f"HIERARCHY CORRECTION: Changed '{text}' from {original_level} to {corrected_level}")
                h['level'] = corrected_level
            corrected_headings.append(h)
        return corrected_headings

    def _extract_headings(self, blocks: List[Dict[str, Any]], toc_page_numbers: set) -> List[Dict[str, Any]]:
        headings = []
        IGNORE_LIST = ['mission statement', 'goals', 'key features', 'contact us', 'summary', 'overview', 'regular pathway', 'distinction pathway', 'revision history']
        
        for b in blocks:
            label, text, page_num = b.get('label'), b.get('text', '').strip(), b.get('page_number')
            if label == 'NONE': continue
            if page_num in toc_page_numbers and 'table of contents' not in text.lower():
                self.logger.info(f"Filtering TOC entry on page {page_num}: '{text}'")
                continue
            if text.lower().rstrip(':') in IGNORE_LIST:
                self.logger.info(f"Filtering out ignored sub-heading: '{text}'")
                continue
            
            headings.append({'text': text, 'level': label, 'page_number': page_num, 'line_position': b['line_position']})
        return sorted(headings, key=lambda x: (x['page_number'], x['line_position']))

    def _deduplicate_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_counts = defaultdict(list)
        for h in headings: text_counts[h['text']].append({'page': h['page_number'], 'y': h['line_position']})
        total_pages = headings[-1]['page_number'] if headings else 1
        common_texts = set()
        for text, occurrences in text_counts.items():
            unique_pages = set(occ['page'] for occ in occurrences)
            if len(unique_pages) > 2 and len(unique_pages) / total_pages > 0.15:
                common_texts.add(text)
                continue
            if len(occurrences) > 1:
                positions = [round(occ['y']) for occ in occurrences]
                if len(set(positions)) < len(positions): common_texts.add(text)
        if common_texts: self.logger.info(f"Removing potential running headers: {list(common_texts)}")
        return [h for h in headings if h['text'] not in common_texts]

    def _extract_title(self, headings: List[Dict[str, Any]]) -> str:
        # This new logic merges consecutive TITLE blocks from the first page
        title_parts = [h['text'] for h in headings if h.get('level') == 'TITLE' and h.get('page_number') == 1]
        if title_parts:
            return " ".join(title_parts)
        # Fallback
        for h in headings:
            if h['level'] == 'H1' and h.get('page_number') == 1: return h['text']
        return "No Title Found"

    def _structure_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{'level': h['level'], 'text': h['text'], 'page': h['page_number']}
                for h in headings if h['level'] != 'TITLE']