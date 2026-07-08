#!/usr/bin/env python3
"""Re-pad numeric image filenames to a fixed width.

e.g. 00070.jpg -> 000070.jpg  (default width 6)

Uses os.scandir + os.rename (no subprocess, no shell glob expansion). Safe
for directories with hundreds of thousands of files.

Examples:
    python pad_frame_names.py                   # cwd, .jpg, width 6
    python pad_frame_names.py -n                # dry run
    python pad_frame_names.py -e png masks/mug  # png masks
    python pad_frame_names.py -v path/to/frames # verbose (echo every rename)
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("dir", nargs="?", default=".",
                   help="Directory to scan (default: current)")
    p.add_argument("-e", "--ext", default="jpg",
                   help="File extension without the dot (default: jpg)")
    p.add_argument("-w", "--width", type=int, default=6,
                   help="Target zero-padding width (default: 6)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Print intended renames but don't actually move files")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print every rename (default: only the final summary)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"error: '{args.dir}' is not a directory")
    if args.width < 1:
        sys.exit(f"error: --width must be >= 1, got {args.width}")
    if not args.ext:
        sys.exit("error: --ext cannot be empty")

    suffix = "." + args.ext.lstrip(".")
    suffix_len = len(suffix)

    n_renamed = 0
    n_already = 0
    n_skipped = 0
    n_seen = 0

    # First pass: collect (src, dst) so we can detect collisions before any
    # rename actually happens. Each entry is just two short strings; cheap.
    pending = []
    with os.scandir(args.dir) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name
            if not name.endswith(suffix):
                continue
            n_seen += 1
            stem = name[:-suffix_len]
            if not stem.isdigit():
                n_skipped += 1
                continue
            num = int(stem)  # base 10, no octal pitfalls
            new_name = f"{num:0{args.width}d}{suffix}"
            if new_name == name:
                n_already += 1
                continue
            pending.append((name, new_name))

    # Pre-flight collision check.
    # A target collides if:
    #   (a) it's another file that already exists on disk and isn't itself
    #       being renamed away, or
    #   (b) two source files map to the same target.
    srcs = {src for src, _ in pending}
    seen_targets = {}
    existing = set(os.listdir(args.dir))
    for src, dst in pending:
        if dst in seen_targets:
            sys.exit(f"error: two sources map to the same target '{dst}': "
                     f"'{seen_targets[dst]}' and '{src}'")
        seen_targets[dst] = src
        if dst in existing and dst not in srcs:
            sys.exit(f"error: target '{dst}' already exists "
                     f"(would clobber rename of '{src}')")

    # Apply renames.
    for src, dst in pending:
        src_path = os.path.join(args.dir, src)
        dst_path = os.path.join(args.dir, dst)
        if args.dry_run:
            if args.verbose:
                print(f"would: {src} -> {dst}")
        else:
            os.rename(src_path, dst_path)
            if args.verbose:
                print(f"{src} -> {dst}")
        n_renamed += 1

    print(f"seen: {n_seen}   renamed: {n_renamed}   "
          f"already-correct: {n_already}   skipped: {n_skipped}")
    if args.dry_run:
        print("(dry run; no files were actually moved)")


if __name__ == "__main__":
    main()
