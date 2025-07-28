import re
import numpy as np
from typing import List, Dict, Any
import logging

class FeatureExtractor:
    """Extracts a comprehensive set of features for high-accuracy heading classification."""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def extract_features(self, blocks: List[Dict[str, Any]]) -> np.ndarray:
        if not blocks:
            return np.array([])
        
        features = []
        
        page_font_sizes = [b.get('font_size', 0) for b in blocks]
        avg_fs = np.mean(page_font_sizes) if page_font_sizes else 0
        max_fs = np.max(page_font_sizes) if page_font_sizes else 0
        min_fs = np.min(page_font_sizes) if page_font_sizes else 0
        std_fs = np.std(page_font_sizes) if page_font_sizes else 0
        
        for i, block in enumerate(blocks):
            features.append(self._extract_block_features(
                block, avg_fs, max_fs, min_fs, std_fs, i, blocks
            ))
        return np.array(features)
    
    def _extract_block_features(self, block, avg_fs, max_fs, min_fs, std_fs, index, all_blocks):
        text = block.get('text', '')
        font_size = block.get('font_size', 0)
        word_count = len(text.split())

        # Font and Style Features
        font_size_ratio = font_size / avg_fs if avg_fs > 0 else 1.0
        font_size_normalized = (font_size - min_fs) / (max_fs - min_fs) if max_fs > min_fs else 0
        font_size_z_score = (font_size - avg_fs) / std_fs if std_fs > 0 else 0
        is_bold = float(block.get('is_bold', False))
        is_italic = float(block.get('is_italic', False))

        # Content and Text Features
        text_length = len(text)
        char_count = len(text.replace(' ', ''))
        
        # Positional Features
        relative_x = block.get('relative_x', 0)
        relative_y = block.get('relative_y', 0)
        is_near_top = float(relative_y < 0.15)
        is_centered = float(0.3 < relative_x < 0.7 and word_count < 10)

        # Pattern Features
        is_numbered = float(bool(re.match(r'^\d+\.', text.strip())))
        is_hierarchical = float(bool(re.match(r'^\d+\.\d+', text.strip())))
        is_all_caps = float(text.isupper() and word_count > 1 and text_length > 5)
        is_title_case = float(text.istitle() and word_count > 1)
        
        # Punctuation Features
        ends_with_colon = float(text.strip().endswith(':'))

        # Contextual and Structural Features
        is_short_line = float(word_count < 8)
        is_standalone = float(text_length < 100)
        spacing_before = self._get_spacing_before(block, all_blocks, index)
        spacing_after = self._get_spacing_after(block, all_blocks, index)
        is_isolated = float(spacing_before > (font_size * 1.5) and spacing_after > (font_size * 1.5))
        
        # Data Source Feature (CRITICAL)
        is_low_fidelity = 1.0 if block.get('source') == 'digital_search' else 0.0

        feature_vector = [
            font_size, font_size_ratio, font_size_normalized, font_size_z_score,
            is_bold, is_italic,
            text_length, char_count, word_count,
            relative_x, relative_y, is_near_top, is_centered,
            is_numbered, is_hierarchical, is_all_caps, is_title_case,
            ends_with_colon,
            is_short_line, is_standalone, is_isolated,
            spacing_before, spacing_after,
            is_low_fidelity
        ]
        return feature_vector

    def _get_spacing_before(self, b, all_b, i):
        if i == 0 or all_b[i - 1].get('page_number') != b.get('page_number'):
            return 50.0
        prev_block_bottom = all_b[i - 1].get('bbox', {}).get('y1', b.get('line_position', 0))
        return b.get('line_position', 0) - prev_block_bottom
    
    def _get_spacing_after(self, b, all_b, i):
        if i >= len(all_b) - 1 or all_b[i + 1].get('page_number') != b.get('page_number'):
            return 50.0
        next_block_top = all_b[i + 1].get('line_position', b.get('bbox', {}).get('y1', 0))
        return next_block_top - b.get('bbox', {}).get('y1', 0)

    def get_feature_names(self) -> List[str]:
        return [
            'font_size', 'font_size_ratio', 'font_size_normalized', 'font_size_z_score', 'is_bold', 'is_italic',
            'text_length', 'char_count', 'word_count', 'relative_x', 'relative_y', 'is_near_top', 'is_centered',
            'is_numbered', 'is_hierarchical', 'is_all_caps', 'is_title_case', 'ends_with_colon',
            'is_short_line', 'is_standalone', 'spacing_before', 'spacing_after', 'is_isolated', 'is_low_fidelity'
        ]