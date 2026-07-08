#!/usr/bin/env bash
# Re-pad numeric image filenames to a fixed width.
# e.g. 00070.jpg -> 000070.jpg  (default width 6)
#
# Streaming + zero-subshell version: safe for directories with tens of
# thousands of files. No per-file echo by default; pass -v for verbose.
#
# Usage:
#   ./pad_frame_names.sh                      # default: cwd, .jpg, width 6
#   ./pad_frame_names.sh -n                   # dry run
#   ./pad_frame_names.sh -w 6 -e png .        # png masks
#   ./pad_frame_names.sh -v path/to/frames    # verbose
#
# Tips:
#   - Always run with -n first on a large dir.
#   - If your dir has >10k files, run inside tmux/screen so a flaky SSH
#     connection can't strand the rename.

set -euo pipefail

dry_run=0
verbose=0
ext="jpg"
width=6

while getopts ":nve:w:h" opt; do
  case "$opt" in
    n) dry_run=1 ;;
    v) verbose=1 ;;
    e) ext="$OPTARG" ;;
    w) width="$OPTARG" ;;
    h)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      echo "usage: $0 [-n] [-v] [-e EXT] [-w WIDTH] [DIR]" >&2
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

dir="${1:-.}"
if [ ! -d "$dir" ]; then
  echo "error: '$dir' is not a directory" >&2
  exit 1
fi

n_renamed=0
n_skipped=0
n_already=0

# Stream filenames from `find` (NULL-delimited, no shell glob expansion).
# Parameter expansion is used everywhere; no $(...) or backticks per file.
while IFS= read -r -d '' f; do
  bn="${f##*/}"
  parent="${f%/*}"
  stem="${bn%.${ext}}"

  if ! [[ "$stem" =~ ^[0-9]+$ ]]; then
    n_skipped=$((n_skipped + 1))
    continue
  fi

  # Decimal forcing (10#) avoids octal parsing of leading-zero stems.
  num=$((10#$stem))
  printf -v new "%0${width}d.%s" "$num" "$ext"

  if [ "$bn" = "$new" ]; then
    n_already=$((n_already + 1))
    continue
  fi

  newpath="$parent/$new"
  if [ -e "$newpath" ]; then
    echo "ERROR: target '$newpath' already exists (collision with '$f')" >&2
    exit 1
  fi

  if [ "$dry_run" -eq 1 ]; then
    [ "$verbose" -eq 1 ] && printf 'would: %s -> %s\n' "$bn" "$new"
  else
    mv -- "$f" "$newpath"
    [ "$verbose" -eq 1 ] && printf '%s -> %s\n' "$bn" "$new"
  fi
  n_renamed=$((n_renamed + 1))
done < <(find "$dir" -maxdepth 1 -type f -name "*.${ext}" -print0)

echo "renamed: $n_renamed   already-correct: $n_already   skipped: $n_skipped"
[ "$dry_run" -eq 1 ] && echo "(dry run; no files were actually moved)"
