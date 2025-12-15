#!/usr/bin/env python3
"""
Production OCR Text Extractor with Elasticsearch Integration
Optimized for high-throughput batch processing with 2-tile splitting
"""

from paddleocr import PaddleOCR
import numpy as np
from numba import jit
import sys
import os
from pdf2image import convert_from_path
import argparse
from tqdm import tqdm
import gc
import cv2
from PIL import Image
import re

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None
ES_URL = 'http://localhost:9200'


@jit(nopython=True, cache=True)
def normalize_coordinates_batch(coords, width, height):
    """Vectorized coordinate normalization"""
    normalized = np.empty_like(coords)
    w_inv = 1.0 / width
    h_inv = 1.0 / height
    
    for i in range(len(coords)):
        normalized[i, 0] = coords[i, 0] * w_inv
        normalized[i, 1] = coords[i, 1] * h_inv
        normalized[i, 2] = coords[i, 2] * w_inv
        normalized[i, 3] = coords[i, 3] * h_inv
    return normalized


class TextDetection():
    def __init__(self) -> None:
        self.ocr = None
        self.stats = {'files': 0, 'pages': 0, 'words': 0}

    def initPaddle(self, det_limit=1920) -> None:
        """Initialize PaddleOCR"""
        self.ocr = PaddleOCR(
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_type='max',
            text_det_limit_side_len=det_limit,
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_recognition_batch_size=16,
        )

    def split_into_tiles(self, img, num_tiles=2, overlap=50):
        """Split image into tiles"""
        h, w = img.shape[:2]
        
        if num_tiles == 1:
            return [(img, 0, 0)]
        elif num_tiles == 2:
            mid_y = h // 2
            return [
                (img[0:mid_y + overlap, :], 0, 0),
                (img[mid_y - overlap:h, :], 0, mid_y - overlap)
            ]
        elif num_tiles == 4:
            mid_y, mid_x = h // 2, w // 2
            return [
                (img[0:mid_y + overlap, 0:mid_x + overlap], 0, 0),
                (img[0:mid_y + overlap, mid_x - overlap:w], mid_x - overlap, 0),
                (img[mid_y - overlap:h, 0:mid_x + overlap], 0, mid_y - overlap),
                (img[mid_y - overlap:h, mid_x - overlap:w], mid_x - overlap, mid_y - overlap)
            ]
        else:
            raise ValueError(f"num_tiles must be 1, 2 or 4")

    def process_tile(self, tile_data):
        """Process single tile"""
        tile, offset_x, offset_y, tile_idx = tile_data
        result = self.ocr.predict(tile)
        return result, offset_x, offset_y, tile_idx

    def adjust_coordinates(self, poly, offset_x, offset_y):
        """Adjust coordinates to full image space"""
        return [[float(pt[0]) + offset_x, float(pt[1]) + offset_y] for pt in poly]

    def merge_results(self, tile_results, overlap, num_tiles):
        """Merge results from tiles"""
        merged = []
        
        for result, offset_x, offset_y, tile_idx in tile_results:
            if not result or not result[0]:
                continue
            
            for res in result:
                texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
                scores = res.get('rec_scores', []) if isinstance(res, dict) else getattr(res, 'rec_scores', [])
                polys = res.get('rec_polys', []) if isinstance(res, dict) else getattr(res, 'rec_polys', [])
                
                if not texts:
                    continue
                
                for j, text in enumerate(texts):
                    if not text or not str(text).strip():
                        continue
                    
                    score = scores[j] if j < len(scores) else 0.0
                    poly = polys[j] if j < len(polys) else None
                    
                    if poly is None:
                        continue
                    
                    adjusted_box = self.adjust_coordinates(poly, offset_x, offset_y)
                    
                    # Skip overlap regions
                    if num_tiles > 1 and tile_idx > 0:
                        center_y = sum(pt[1] for pt in adjusted_box) / 4
                        if num_tiles == 2 and tile_idx == 1 and center_y < offset_y + overlap:
                            continue
                        if num_tiles == 4:
                            center_x = sum(pt[0] for pt in adjusted_box) / 4
                            if tile_idx == 1 and center_x < offset_x + overlap:
                                continue
                            if tile_idx == 2 and center_y < offset_y + overlap:
                                continue
                            if tile_idx == 3 and (center_x < offset_x + overlap or center_y < offset_y + overlap):
                                continue
                    
                    merged.append({
                        "text": str(text),
                        "confidence": float(score),
                        "box": adjusted_box,
                        "tile_idx": tile_idx
                    })
        
        return merged

    def paddleOCR(self, file_path, num_tiles=2, dpi=200):
        """Main OCR processing"""
        images = convert_from_path(file_path, dpi=dpi)
        img = np.array(images[0])
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        tiles = self.split_into_tiles(img_bgr, num_tiles, overlap=50)
        tile_data = [(tile, ox, oy, i) for i, (tile, ox, oy) in enumerate(tiles)]
        
        tile_results = [self.process_tile(td) for td in tile_data]
        merged_results = self.merge_results(tile_results, 50, num_tiles)

        self.stats['pages'] += 1
        self.stats['words'] += len(merged_results)

        img_height, img_width = img.shape[:2]

        del images, img, img_bgr, tiles, tile_data, tile_results
        gc.collect()

        return merged_results, img_width, img_height


# --- ELASTICSEARCH ---

def get_index_name(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    return folder_name.lower().replace(' ', '_').replace('-', '_')


def setup_elasticsearch_index(es, index_name):
    all_indices = es.indices.get_alias(index="*")
    user_indices = [idx for idx in all_indices.keys() if not idx.startswith('.')]
    
    if user_indices:
        for idx in user_indices:
            es.indices.delete(index=idx)
    
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "analysis": {
                "normalizer": {
                    "lowercase_norm": {"type": "custom", "filter": ["lowercase"]}
                }
            }
        },
        "mappings": {
            "properties": {
                "folder_name": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "source_folder": {"type": "keyword"},
                "text_source": {"type": "keyword"},
                "image_info": {
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "dpi": {"type": "integer"},
                        "path": {"type": "keyword"}
                    }
                },
                "words": {
                    "type": "nested",
                    "properties": {
                        "word": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "normalizer": "lowercase_norm",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "confidence": {"type": "float"},
                        "rotation": {"type": "integer"},
                        "coordinates": {
                            "properties": {
                                "x0": {"type": "float"},
                                "y0": {"type": "float"},
                                "x1": {"type": "float"},
                                "y1": {"type": "float"}
                            }
                        }
                    }
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)


def bulk_index_to_elasticsearch(es, index_name, pages_data, bulk_size=500):
    if not pages_data:
        return 0, 0
    
    actions = [
        {'_index': index_name, '_id': page['_id'], '_source': page['_source']}
        for page in pages_data
    ]
    
    success, failed = bulk(
        es.options(request_timeout=60),
        actions,
        chunk_size=bulk_size,
        raise_on_error=False
    )
    
    del actions
    gc.collect()
    
    return success, len(failed) if failed else 0


def convert_to_elasticsearch_format(file_path, merged_results, index_name, img_width, img_height, dpi=200):
    file_name = os.path.splitext(os.path.basename(file_path))[0].lower()
    doc_id = f"{file_name}_page_1"
    
    word_docs = []
    
    if merged_results:
        coords = np.empty((len(merged_results), 4), dtype=np.float32)
        valid_results = []
        
        for detection in merged_results:
            text = detection.get("text", "")
            confidence = detection.get("confidence", 0.0)
            box = detection.get("box", [])
            
            if not text or not box or len(box) < 4:
                continue
            
            text_stripped = text.strip()
            if len(text_stripped) == 0:
                continue
                        
            if re.fullmatch(r'[a-zA-Z]{1,2}', text_stripped):
                continue
            
            if re.fullmatch(r'\d', text_stripped):
                continue
            
            valid_results.append((text_stripped, confidence))
            
            coords[len(valid_results)-1] = [
                min(box[0][0], box[3][0]),
                min(box[0][1], box[1][1]),
                max(box[1][0], box[2][0]),
                max(box[2][1], box[3][1])
            ]
        
        coords = coords[:len(valid_results)]
        normalized_coords = normalize_coordinates_batch(coords, img_width, img_height)
        
        for i, (text, confidence) in enumerate(valid_results):
            word_docs.append({
                "word": text,
                "confidence": float(confidence),
                "rotation": 0,
                "coordinates": {
                    "x0": float(normalized_coords[i, 0]),
                    "y0": float(normalized_coords[i, 1]),
                    "x1": float(normalized_coords[i, 2]),
                    "y1": float(normalized_coords[i, 3])
                }
            })
    
    return {
        "_id": doc_id,
        "_source": {
            "folder_name": index_name,
            "source_folder": os.path.basename(os.path.dirname(file_path)),
            "file_name": file_name,
            "page_number": 1,
            "text_source": "paddleocr",
            "image_info": {
                "width": img_width,
                "height": img_height,
                "dpi": dpi,
                "path": f"{file_name}_page_1.png"
            },
            "words": word_docs
        }
    }


# --- RENDER ONLY ---

def render_pdfs_to_images(folder_path, output_dir, dpi=200):
    """
    Render all PDFs in folder to PNG images.
    Naming convention: {filename}_page_{page_number}.png
    """
    # Create output directory in main sisint_docsearch folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"📁 Output directory: {output_path}")
    
    # Find all PDFs
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        pdf_files.extend(
            os.path.join(root, f)
            for f in files
            if f.lower().endswith('.pdf') and not f.endswith('.Zone.Identifier')
        )
    
    if not pdf_files:
        print(f"❌ No PDFs found in {folder_path}")
        return 0
    
    pdf_files.sort()
    print(f"📄 Found {len(pdf_files)} PDFs to render")
    
    total_pages = 0
    
    for pdf_path in tqdm(pdf_files, desc="Rendering PDFs"):
        try:
            # Get filename without extension (lowercase)
            file_name = os.path.splitext(os.path.basename(pdf_path))[0].lower()
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            for page_num, img in enumerate(images, 1):
                # Naming convention: {filename}_page_{page_number}.png
                output_filename = f"{file_name}_page_{page_num}.png"
                output_filepath = os.path.join(output_path, output_filename)
                
                # Save as PNG
                img.save(output_filepath, 'PNG')
                total_pages += 1
            
            # Cleanup
            del images
            gc.collect()
            
        except Exception as e:
            print(f"\n❌ Error rendering {pdf_path}: {e}")
            continue
    
    print(f"\n✅ Rendered {total_pages} pages from {len(pdf_files)} PDFs")
    print(f"📁 Images saved to: {output_path}")
    return total_pages


# --- CLI ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Production OCR with Elasticsearch")
    
    parser.add_argument("folder_path", help="Path to PDF folder")
    parser.add_argument("--tiles", type=int, choices=[1, 2, 4], default=2,
                        help="Number of tiles (default: 2)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI (default: 200)")
    parser.add_argument("--det-limit", type=int, default=1920, help="Detection limit")
    parser.add_argument("--index", action="store_true", help="Enable Elasticsearch")
    parser.add_argument("--reset", action="store_true", help="Reset ES indices")
    parser.add_argument("--bulk-size", type=int, default=500, help="Bulk size")
    parser.add_argument("--es-url", type=str, default=ES_URL, help="ES URL")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--render-only", action="store_true", 
                        help="Only render PDFs to PNG images in rendered_pages folder (no OCR)")
    parser.add_argument("--output-dir", type=str, default="rendered_pages",
                        help="Output directory for rendered images (default: rendered_pages)")
    
    args = parser.parse_args()
    
    if args.index and not args.reset:
        parser.error("--reset required with --index")
    
    if args.index and not ES_AVAILABLE:
        parser.error("Elasticsearch not installed")
    
    if args.render_only and args.index:
        parser.error("--render-only and --index cannot be used together")
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.isdir(args.folder_path):
        print(f"❌ Invalid directory: {args.folder_path}")
        sys.exit(1)
    
    # Handle render-only mode
    if args.render_only:
        print(f"🖼️  Render-only mode - Converting PDFs to PNG images")
        pages = render_pdfs_to_images(args.folder_path, args.output_dir, args.dpi)
        if pages > 0:
            print("✅ Render complete!")
        sys.exit(0)
    
    # Find PDFs
    pdf_files = []
    for root, dirs, files in os.walk(args.folder_path):
        source = os.path.basename(root) if root != args.folder_path else os.path.basename(args.folder_path)
        pdf_files.extend(
            (os.path.join(root, f), source)
            for f in files
            if f.lower().endswith('.pdf') and not f.endswith('.Zone.Identifier')
        )
    
    if not pdf_files:
        print(f"❌ No PDFs found")
        sys.exit(1)
    
    pdf_files.sort()
    print(f"📄 Found {len(pdf_files)} PDFs")
    
    # Initialize
    te = TextDetection()
    te.initPaddle(det_limit=args.det_limit)
    
    # Setup ES
    es = None
    index_name = None
    all_pages = []
    
    if args.index:
        try:
            es = Elasticsearch(args.es_url, request_timeout=60)
            es.info()
            index_name = get_index_name(args.folder_path)
            setup_elasticsearch_index(es, index_name)
        except Exception as e:
            print(f"❌ ES failed: {e}")
            sys.exit(1)
    
    # Process
    total_stats = {'files': 0, 'pages': 0, 'words': 0, 'indexed': 0, 'failed': 0}
    
    for idx, (file_path, source) in enumerate(tqdm(pdf_files, desc="Processing"), 1):
        try:
            merged_results, img_width, img_height = te.paddleOCR(
                file_path,
                num_tiles=args.tiles,
                dpi=args.dpi
            )
            
            total_stats['files'] += 1
            total_stats['pages'] += 1
            total_stats['words'] += len(merged_results)
            
            if args.index and es and merged_results:
                es_doc = convert_to_elasticsearch_format(
                    file_path, merged_results, index_name,
                    img_width, img_height, args.dpi
                )
                all_pages.append(es_doc)
            
            if args.index and (len(all_pages) >= args.bulk_size or idx == len(pdf_files)):
                if all_pages:
                    success, failed = bulk_index_to_elasticsearch(es, index_name, all_pages, args.bulk_size)
                    total_stats['indexed'] += success
                    total_stats['failed'] += failed
                    all_pages = []
                    
                    if idx % 5 == 0:
                        es.indices.refresh(index=index_name)
            
            if idx % 5 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Finalize
    if args.index and es:
        es.indices.put_settings(
            index=index_name,
            body={"index": {"refresh_interval": "1s", "number_of_replicas": 1}}
        )
        es.indices.refresh(index=index_name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Files:     {total_stats['files']}")
    print(f"Pages:     {total_stats['pages']}")
    print(f"Words:     {total_stats['words']:,}")
    if args.index:
        print(f"Indexed:   {total_stats['indexed']}")
    print(f"{'='*60}\n")
    print("✅ Complete!")