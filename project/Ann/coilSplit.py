#!/usr/bin/env python3

"""
COIL-100 Dataset Splitter.
Splits dataset by rotation angles into train/validation/test sets.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from pdb import set_trace


def parse_filename(filename):
    """Extract object number and degree from filename.

    Args:
        filename: String like 'obj1__0.jpg'

    Returns:
        tuple: (object_num, degree) or None if parsing fails
    """
    match = re.match(r'obj(\d+)__(\d+)\.png', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def split_dataset(source_dir, train_dir, validation_dir, test_dir):
    """Split COIL-100 dataset by rotation angles.

    Split strategy:
    - Train: 0° - 250° (50 images per object)
    - Validation: 255° - 305° (11 images per object)
    - Test: 310° - 355° (11 images per object)
    """
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    validation_path = Path(validation_dir)
    test_path = Path(test_dir)

    set_trace()
    # Create output directories
    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Verify source directory exists
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Process all images
    stats = {'train': 0, 'validation': 0, 'test': 0, 'skipped': 0}

    for img_file in sorted(source_path.glob('obj*__*.png')):
        parsed = parse_filename(img_file.name)

        if parsed is None:
            stats['skipped'] += 1
            continue

        obj_num, degree = parsed

        # Determine destination based on degree
        if 0 <= degree <= 250:
            dest_dir = train_path
            stats['train'] += 1
        elif 255 <= degree <= 305:
            dest_dir = validation_path
            stats['validation'] += 1
        elif 310 <= degree <= 355:
            dest_dir = test_path
            stats['test'] += 1
        else:
            # Handle edge cases (gaps in 5° increments)
            stats['skipped'] += 1
            continue

        # Copy file to destination
        shutil.copy2(img_file, dest_dir / img_file.name)

    # Print statistics
    print(f"# Split completed successfully!", file=sys.stderr)
    print(f"# Train: {stats['train']} images", file=sys.stderr)
    print(f"# Validation: {stats['validation']} images", file=sys.stderr)
    print(f"# Test: {stats['test']} images", file=sys.stderr)
    if stats['skipped'] > 0:
        print(f"# Skipped: {stats['skipped']} files", file=sys.stderr)
    print(file=sys.stderr)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Split COIL-100 dataset into train/validation/test sets by rotation angle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  coilSplit -source /tmp/coil100_images -train /tmp/train -test /tmp/test -validation /tmp/validation

Split strategy:
  Train:      0° - 250°  (~50 images per object)
  Validation: 255° - 305° (~11 images per object)
  Test:       310° - 355° (~11 images per object)
        """
    )

    parser.add_argument('-source', required=True, help='Source directory with COIL-100 images')
    parser.add_argument('-train', required=True, help='Output directory for training set')
    parser.add_argument('-validation', required=True, help='Output directory for validation set')
    parser.add_argument('-test', required=True, help='Output directory for test set')

    args = parser.parse_args()

    # Perform the split
    split_dataset(args.source, args.train, args.test, args.validation)

    # Output environment variable exports
    print(f'export COIL100_SOURCE="{args.source}"')
    print(f'export COIL100_TRAIN="{args.train}"')
    print(f'export COIL100_TEST="{args.test}"')
    print(f'export COIL100_VALIDATION="{args.validation}"')


if __name__ == '__main__':
    main()
