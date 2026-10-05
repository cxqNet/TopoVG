#!/usr/bin/env python3
"""Build the TopoVG test JSON from original RRSIS-D test refs and XML boxes.
"""

import argparse
import json
import math
import os
import pickle
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip()).strip(" \t\r\n\"'").lower()


def compact_number(value):
    return int(value) if value.is_integer() else value


def sentence_texts(sentence):
    if not isinstance(sentence, dict):
        raise ValueError("sentence must be a dictionary containing sent/raw")
    texts = [sentence.get(key) for key in ("sent", "raw")]
    texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError("sentence has no valid sent/raw expression")
    return texts


def read_xml(path, file_name):
    root = ET.parse(path).getroot()
    xml_name = root.findtext("filename")
    if xml_name and Path(xml_name.strip()).name != file_name:
        raise ValueError(
            f"XML filename={xml_name!r} does not match ref file_name={file_name!r}"
        )
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    depth = int(root.findtext("size/depth", "3"))
    if min(width, height, depth) <= 0:
        raise ValueError("Invalid XML image dimensions")
    objects = root.findall("object")
    descriptions = [normalize(obj.findtext("description", "")) for obj in objects]
    return width, height, depth, objects, descriptions


def build_samples(refs, ann_dir, data_root, expected_samples):
    if not isinstance(refs, (list, tuple)) or any(not isinstance(r, dict) for r in refs):
        raise ValueError("refs pickle must contain a list or tuple of ref dictionaries")
    selected = [r for r in refs if r.get("split") == "test"]
    if not selected:
        raise ValueError("No entries with split='test' found in refs")
    samples, errors, xml_cache = [], [], {}
    sentence_count = 0
    sentence_positions = []
    for ref_index, ref in enumerate(selected, 1):
        sentences = ref.get("sentences")
        label = f"test ref #{ref_index}, ref_id={ref.get('ref_id')}, file={ref.get('file_name')}"
        if not isinstance(sentences, (list, tuple)) or not sentences:
            errors.append(f"{label}: sentences must be a nonempty list or tuple")
            continue
        sentence_count += len(sentences)
        for sentence_index, sentence in enumerate(sentences):
            try:
                file_name = ref.get("file_name")
                if not isinstance(file_name, str) or not file_name:
                    raise ValueError("ref has no valid file_name")
                if Path(file_name).name != file_name or "\\" in file_name:
                    raise ValueError("file_name must be an image filename without a directory")
                image = Path(file_name).stem
                if not re.fullmatch(r"[A-Za-z0-9_-]+", image):
                    raise ValueError(f"Invalid image stem for feature filenames: {image!r}")
                # Match evaluate.py's image lookup.
                image_paths = [
                    data_root / "JPEGImages" / (image + ext)
                    for ext in (".jpg", ".png", ".jpeg")
                ]
                found = [p for p in image_paths if p.is_file()]
                if len(found) != 1 or found[0].name != file_name:
                    raise ValueError(
                        f"Expected exactly one {file_name} in JPEGImages; found: {found}"
                    )
                texts = sentence_texts(sentence)
                xml_path = ann_dir / (image + ".xml")
                if xml_path not in xml_cache:
                    xml_cache[xml_path] = read_xml(xml_path, file_name)
                width, height, depth, objects, descriptions = xml_cache[xml_path]
                keys = {normalize(text) for text in texts}
                matches = [i for i, desc in enumerate(descriptions) if desc in keys]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expression {texts[0]!r} matched {len(matches)} objects in {xml_path}; "
                        f"expected exactly one match. Candidate indices: {matches}"
                    )
                object_index = matches[0]
                obj = objects[object_index]
                category = (obj.findtext("name") or "").strip()
                if not category:
                    raise ValueError("Matched object has no category name")
                box = [
                    float(obj.findtext("bndbox/" + key, "nan"))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                ]
                x1, y1, x2, y2 = box
                if not (
                    all(math.isfinite(v) for v in box)
                    and 0 <= x1 < x2 <= width
                    and 0 <= y1 < y2 <= height
                ):
                    raise ValueError(f"Invalid bbox or coordinates outside {width}x{height}: {box}")
                object_id = obj.findtext("id")
                samples.append({
                    "id": f"{image}_{object_index:03d}",
                    "image": image,
                    "width": width,
                    "height": height,
                    "category": category,
                    "expression": texts[0],
                    "bbox_xyxy": [compact_number(v) for v in box],
                    "bbox_xywh": [
                        compact_number(v) for v in (x1, y1, x2-x1, y2-y1)
                    ],
                    "object_index": object_index,
                    "depth": depth,
                    "source_object_id": object_id.strip() if object_id else None,
                })
                sentence_positions.append((ref_index, sentence_index))
            except (ValueError, TypeError, OSError, ET.ParseError) as exc:
                errors.append(f"{label}, sentence #{sentence_index}: {exc}")
    if expected_samples and sentence_count != expected_samples:
        errors.append(
            f"Test refs contain {sentence_count} expressions; expected {expected_samples}. "
            "Verify the refs version or use --expected-samples 0 to disable this check."
        )
    if errors:
        raise ValueError(
            f"Validation failed with {len(errors)} errors; JSON was not written.\n"
            + "\n".join(errors[:20])
            + ("\nAdditional errors omitted." if len(errors) > 20 else "")
        )
    if len(samples) != sentence_count:
        raise ValueError("Generated sample count does not match the refs expression count")
    # Preserve existing IDs; disambiguate repeated references to the same object.
    base_counts = Counter(s["id"] for s in samples)
    for sample, (ref_index, sentence_index) in zip(samples, sentence_positions):
        if base_counts[sample["id"]] > 1:
            sample["id"] += f"_r{ref_index:05d}_s{sentence_index:03d}"
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate generated sample IDs")
    return selected, samples


def check_parser(samples, path):
    with path.open(encoding="utf-8") as handle:
        trees = json.load(handle)
    if not isinstance(trees, dict):
        raise ValueError("Parser JSON must be an expression-to-graph dictionary")
    lookup = {}
    for text, graph in trees.items():
        key = normalize(text)
        if key in lookup and lookup[key] != graph:
            raise ValueError(f"Conflicting normalized parser key: {text!r}")
        lookup[key] = graph
    missing = []
    for sample in samples:
        graph = trees.get(
            sample["expression"],
            lookup.get(normalize(sample["expression"])),
        )
        if not isinstance(graph, dict) or not isinstance(graph.get("target"), dict):
            missing.append((sample["id"], sample["expression"]))
    if missing:
        raise ValueError(
            f"{len(missing)} samples have missing or invalid parser entries; "
            f"first five: {missing[:5]}"
        )


def save_with_backup(samples, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(samples, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=output.name + ".",
        suffix=".tmp",
        dir=output.parent,
    )
    backup = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = output.with_name(output.name + "." + stamp + ".bak")
            shutil.copy2(output, backup)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--refs", required=True, help="Original refs(unc).p")
    cli.add_argument("--ann-dir", required=True, help="Original ann_split XML directory")
    cli.add_argument("--data-root", required=True, help="TopoVG data root containing JPEGImages")
    cli.add_argument("--output-json", help="Default: data-root/split/test.json; backup existing output")
    cli.add_argument("--parser-json", help="Optional parser coverage check before writing")
    cli.add_argument("--expected-samples", type=int, default=3481, help="Default 3481; 0 disables")
    args = cli.parse_args()

    def resolve(value):
        return Path(value).expanduser().resolve()

    try:
        source, ann_dir, data_root = resolve(args.refs), resolve(args.ann_dir), resolve(args.data_root)
        output = resolve(args.output_json) if args.output_json else data_root / "split/test.json"
        parser_path = resolve(args.parser_json) if args.parser_json else None
        if args.expected_samples < 0:
            raise ValueError("--expected-samples must be nonnegative")
        if not ann_dir.is_dir() or not (data_root / "JPEGImages").is_dir():
            raise ValueError("ann-dir or data-root/JPEGImages does not exist")
        if output == source or (parser_path and output == parser_path) or output.suffix.lower() != ".json":
            raise ValueError("Output must be a JSON file and must not overwrite refs or parser")
        with source.open("rb") as handle:
            refs = pickle.load(handle, encoding="latin1")
        selected, samples = build_samples(refs, ann_dir, data_root, args.expected_samples)
        if parser_path:
            check_parser(samples, parser_path)
        backup = save_with_backup(samples, output)
    except (ValueError, TypeError, OSError, EOFError, pickle.UnpicklingError) as exc:
        cli.exit(1, f"Error: {exc}\n")

    print(f"Refs: {source}")
    print(f"XML annotations: {ann_dir}")
    print(f"Test refs: {len(selected)}")
    print(f"Test samples: {len(samples)}")
    print(f"Unique images: {len({s['image'] for s in samples})}")
    print(f"Unique expressions: {len({normalize(s['expression']) for s in samples})}")
    if parser_path:
        print(f"Parser coverage: {len(samples)}/{len(samples)}")
    if backup:
        print(f"Backup saved to: {backup}")
    print(f"Test JSON saved to: {output}")


if __name__ == "__main__":
    main()