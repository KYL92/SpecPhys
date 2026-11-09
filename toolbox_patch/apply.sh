#!/bin/bash
# Apply the SpecPhys modifications to rPPG-Toolbox (Responsible AI License v1.1, see LICENSE-RAIL.txt).
# Usage: bash toolbox_patch/apply.sh [TOOLBOX_DIR]   (default: repos/rPPG-Toolbox next to this repository's root)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); TB=${1:-$HERE/../repos/rPPG-Toolbox}
BASE=b7500b848f84ad7f86e277b4612563b69f4f88f9
[ -d "$TB/.git" ] || git clone https://github.com/ubicomplab/rPPG-Toolbox "$TB"
git -C "$TB" checkout -q "$BASE"
git -C "$TB" apply --whitespace=nowarn "$HERE/rppg_toolbox.patch"
(cd "$HERE/new_files" && find . -name '*.py' | while read -r f; do install -D -m 644 "$f" "$TB/$f"; done)
echo "rPPG-Toolbox at $BASE with the SpecPhys modifications: $TB"
