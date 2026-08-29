#!/usr/bin/env python3
"""
Exact Sequence Matching Novelty Metric for AnalogGenie with FT

Measures novelty by checking if generated sequences exactly match
any sequence in the training dataset. Any sequence NOT found in
Training.npy is considered novel.

Analyzes single inference folder: Inference/

Metric: Binary classification (Novel vs Memorized)
- Novel: Sequence NOT in Training.npy
- Memorized: Exact match found in Training.npy
"""

import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
import time
import hashlib

# Configuration
BASE_DIR = Path(__file__).parent
_training_renamed = BASE_DIR / 'Training_renamed.npy'
_training_default = BASE_DIR / 'Training.npy'
TRAINING_NPY = _training_renamed if _training_renamed.exists() else _training_default
LOG_FILE = BASE_DIR / 'METRIC_ExactMatching.log'
CACHE_FILE = BASE_DIR / f"{TRAINING_NPY.stem}_ExactMatchingHashes.npy"


def log_message(message):
    """Write a message to both stdout and the log file."""
    print(message)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(message + '\n')
    except Exception:
        # Logging must never be the reason the metric crashes.
        pass


def parse_txt_file(file_path):
    """
    Parse .txt file and extract tokens (excluding TRUNCATE).
    
    Args:
        file_path: Path to .txt file
    
    Returns:
        Tuple of tokens (as tuple for hashing)
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()

    if not content:
        raise ValueError("file is empty")

    if '->' not in content:
        raise ValueError("missing '->' delimiter")
    
    # Split by '->' and remove empty/whitespace
    tokens = [t.strip() for t in content.split('->') if t.strip()]

    if not tokens:
        raise ValueError("no tokens found after splitting")
    
    # Remove TRUNCATE tokens
    tokens = [t for t in tokens if t != 'TRUNCATE']

    if not tokens:
        raise ValueError("only TRUNCATE tokens found")
    
    return tuple(tokens)


def normalize_sequence(seq):
    """
    Normalize a sequence from Training.npy to tuple format.
    
    Args:
        seq: Sequence array from .npy file
        
    Returns:
        Tuple of tokens (excluding TRUNCATE)
    """
    tokens = []
    if seq is None:
        raise ValueError("Encountered None sequence")

    try:
        iterator = iter(seq)
    except TypeError as e:
        raise TypeError(f"Sequence is not iterable: {type(seq).__name__}") from e

    for token in iterator:
        token_str = str(token).strip()
        if token_str == 'TRUNCATE' or token_str == '':
            break
        tokens.append(token_str)
    
    return tuple(tokens)


def hash_sequence(tokens):
    """
    Convert a token sequence into a stable 64-bit hash.

    This is much smaller than storing the full sequence in RAM and is used for
    streaming exact-match lookup.
    """
    joined = '\x1f'.join(tokens).encode('utf-8')
    digest = hashlib.blake2b(joined, digest_size=8).digest()
    return np.frombuffer(digest, dtype=np.uint64)[0]


def load_training_hash_cache(training_npy_path):
    """
    Load the prebuilt hash cache used for exact matching.

    This function is intentionally cache-only so the script can run on machines
    with limited RAM. If the cache is missing, the user must build it on a
    machine with enough memory first.
    """
    log_message(f"Cache file: {CACHE_FILE}")

    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Hash cache not found: {CACHE_FILE}. "
            "This version of METRIC_ExactMatching.py is cache-only and will not "
            "load the full Training.npy/Training_renamed.npy because the file uses "
            "Python object dtype and is too memory-heavy for 8 GB RAM. "
            "Please build the cache on a machine with more memory, then rerun this script."
        )

    try:
        cached = np.load(CACHE_FILE, mmap_mode='r', allow_pickle=False)
        log_message(f"Loaded hash cache with {len(cached)} entries")
        return cached
    except Exception as e:
        raise RuntimeError(
            f"Failed to load hash cache from {CACHE_FILE}: {type(e).__name__}: {e}"
        ) from e


def contains_hash(sorted_hash_array, query_hash):
    """Binary-search membership test on a sorted uint64 hash array."""
    idx = np.searchsorted(sorted_hash_array, query_hash)
    return idx < len(sorted_hash_array) and sorted_hash_array[idx] == query_hash


def build_training_set_index(training_npy_path):
    """
    Load Training.npy and build/load a compact hash cache for O(log n) lookup.

    Returns:
        Sorted numpy array of uint64 hashes.
    """
    log_message(f"\n{'='*70}")
    log_message("Loading Training Data")
    log_message(f"{'='*70}")
    log_message(f"File: {training_npy_path}")
    log_message("Step: checking whether training file exists")

    if not training_npy_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {training_npy_path}. "
            "Expected Training_renamed.npy or Training.npy in the same directory as this script."
        )

    return load_training_hash_cache(training_npy_path)


def analyze_inference_folder(folder_path, training_sequences):
    """
    Analyze inference folder and check novelty.
    
    Args:
        folder_path: Path to inference folder
        training_sequences: Set of training sequence tuples
    
    Returns:
        Dictionary with analysis results
    """
    folder_name = folder_path.name
    log_message(f"\n{'='*70}")
    log_message(f"Analyzing: {folder_name}")
    log_message(f"{'='*70}")
    
    # Collect all .txt files
    txt_files = sorted(folder_path.glob('run*.txt'))
    log_message(f"Found {len(txt_files)} files")
    
    if not txt_files:
        return {
            'folder': folder_name,
            'total': 0,
            'novel': 0,
            'memorized': 0,
            'novelty_rate': 0.0,
            'error': 'No .txt files found'
        }
    
    novel_count = 0
    memorized_count = 0
    error_count = 0
    memorized_examples = []
    
    for txt_file in tqdm(txt_files, desc=f"Processing {folder_name}"):
        try:
            infer_seq = parse_txt_file(txt_file)

            if not infer_seq:
                error_count += 1
                continue

            infer_hash = hash_sequence(infer_seq)
            
            # Check if sequence exists in training set
            if contains_hash(training_sequences, infer_hash):
                memorized_count += 1
                if len(memorized_examples) < 10:  # Save first 10 examples
                    memorized_examples.append({
                        'file': txt_file.name,
                        'sequence_length': len(infer_seq)
                    })
            else:
                novel_count += 1
        
        except Exception as e:
            error_count += 1
            log_message(f"  Error processing {txt_file.name}: {e}")
    
    total = novel_count + memorized_count
    novelty_rate = (novel_count / total * 100) if total > 0 else 0
    
    results = {
        'folder': folder_name,
        'total': total,
        'novel': novel_count,
        'memorized': memorized_count,
        'errors': error_count,
        'novelty_rate': novelty_rate,
        'memorized_examples': memorized_examples
    }
    
    log_message(f"\nResults:")
    log_message(f"  Total analyzed: {total}")
    if total > 0:
        log_message(f"  Novel: {novel_count} ({novel_count/total*100:.1f}%)")
        log_message(f"  Memorized: {memorized_count} ({memorized_count/total*100:.1f}%)")
    else:
        log_message("  Novel: 0 (0.0%)")
        log_message("  Memorized: 0 (0.0%)")
    if error_count > 0:
        log_message(f"  Errors: {error_count}")
    log_message(f"  Novelty Rate: {novelty_rate:.2f}%")
    
    return results


def find_inference_folders(base_dir):
    """
    Auto-detect inference folder structure.

    Two patterns supported:
    1. Single 'Inference/' folder with run*.txt files (AnalogGenie style)
    2. Multiple 'Inference_*/' folders each with run*.txt files (AnalogToBi style)

    Returns:
        List of Path objects to inference folders that contain run*.txt files
    """
    # Pattern 1: single Inference/ with files
    single = base_dir / 'Inference'
    if single.exists() and any(single.glob('run*.txt')):
        return [single]

    # Pattern 2: multiple Inference_*/ folders
    multi = sorted(base_dir.glob('Inference_*/'))
    multi = [p for p in multi if p.is_dir() and any(p.glob('run*.txt'))]
    if multi:
        return multi

    return []


def main():
    start_time = time.time()

    LOG_FILE.write_text("")
    
    log_message("="*70)
    log_message("EXACT SEQUENCE MATCHING NOVELTY METRIC")
    log_message("="*70)
    log_message("\nMethod: Binary classification")
    log_message(f"  Training file: {TRAINING_NPY.name}")
    log_message("  Novel:      Sequence NOT in training set")
    log_message("  Memorized:  Exact match found in training set")
    
    # Load training data
    try:
        training_sequences = build_training_set_index(TRAINING_NPY)
    except Exception as e:
        log_message(f"\nError: {e}")
        return
    
    # Auto-detect inference folders
    inference_folders = find_inference_folders(BASE_DIR)
    if not inference_folders:
        log_message(f"\nError: No inference folders with run*.txt files found in {BASE_DIR}")
        return
    
    log_message(f"\nDetected {len(inference_folders)} inference folder(s)")
    
    # Analyze each folder
    all_results = []
    for folder in inference_folders:
        results = analyze_inference_folder(folder, training_sequences)
        all_results.append(results)
    
    # Aggregate totals
    total_all = sum(r['total'] for r in all_results)
    novel_all = sum(r['novel'] for r in all_results)
    memorized_all = sum(r['memorized'] for r in all_results)
    novelty_rate_all = (novel_all / total_all * 100) if total_all > 0 else 0
    
    # Summary
    log_message(f"\n{'='*70}")
    log_message("SUMMARY - EXACT SEQUENCE MATCHING NOVELTY")
    log_message(f"{'='*70}")
    if len(all_results) > 1:
        for r in all_results:
            log_message(f"  [{r['folder']}] total={r['total']} novel={r['novel']} ({r['novelty_rate']:.1f}%)")
        log_message(f"{'─'*70}")
    log_message(f"Total generated: {total_all}")
    if total_all > 0:
        log_message(f"Novel:           {novel_all} ({novelty_rate_all:.2f}%)")
    else:
        log_message("Novel:           0 (0.00%)")
    if total_all > 0:
        log_message(f"Memorized:       {memorized_all} ({memorized_all/total_all*100:.2f}%)")
    else:
        log_message("Memorized:       0 (0.00%)")
    errors_all = sum(r.get('errors', 0) for r in all_results)
    if errors_all > 0:
        log_message(f"Errors:          {errors_all}")

    valid_total_all = sum(r['total'] for r in all_results if r['total'] > 0)
    if valid_total_all == 0:
        log_message("\nWarning: All inference folders produced 0 valid sequences.")
        log_message("Novelty rate is reported as 0.00% to keep the output safe and avoid divide-by-zero.")
    
    # Save results as JSON
    output_file = BASE_DIR / 'NOVELTY_ExactMatching_Results.json'
    
    summary = {
        'method': 'exact_sequence_matching',
        'description': 'Binary novelty: sequence not in training set = novel',
        'training_data': {
            'file': str(TRAINING_NPY.name),
            'unique_hashes': len(training_sequences),
            'cache_file': str(CACHE_FILE.name)
        },
        'inference_folders': [str(f.name) for f in inference_folders],
        'per_folder_results': all_results,
        'aggregate': {
            'total': total_all,
            'novel': novel_all,
            'memorized': memorized_all,
            'novelty_rate': novelty_rate_all
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'computation_time_seconds': time.time() - start_time
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_message(f"\n{'='*70}")
    log_message("RESULTS SAVED")
    log_message(f"{'='*70}")
    log_message(f"  Output file: {output_file}")
    log_message(f"  Computation time: {time.time() - start_time:.1f}s")
    
    log_message(f"\nExact Sequence Matching Novelty analysis complete")
    log_message(f"Method: Binary classification (novel vs memorized)")
    log_message(f"Training set: {len(training_sequences):,} unique sequences")
    log_message(f"Novelty rate: {novelty_rate_all:.2f}%")


if __name__ == '__main__':
    main()
