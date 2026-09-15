#!/usr/bin/env bash
# Fetch the pinned upstream checkouts this bridge builds against.
# Idempotent: safe to re-run; existing checkouts are moved to the pin.
# Upstream code is never vendored into this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$(dirname "$ROOT")/vendor}"

pin() { # name, repo url, sha (short or full)
    local name="$1" url="$2" sha="$3" dir="$DEST/$1" full
    if [ -d "$dir/.git" ]; then
        git -C "$dir" fetch -q origin
    else
        git clone -q "$url" "$dir"
    fi
    git -C "$dir" checkout -q "$sha"
    full="$(git -C "$dir" rev-parse "$sha")"
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$full" ]; then
        echo "ERROR: $name is not at $sha" >&2
        exit 1
    fi
    echo "$name @ $(git -C "$dir" rev-parse --short HEAD) -> $dir"
}

mkdir -p "$DEST"
pin GraphGen https://github.com/InternScience/GraphGen 3a3eb097
pin toolgrad https://github.com/zhongyi-zhou/toolgrad c9544f84

# Editable installs so `import toolgrad` resolves. GraphGen's full deps are
# heavy (torch etc.); the bridge only reads its networkx GraphML output, so
# a GraphGen install failure is a warning, not an error.
python3 -m pip install -q -e "$DEST/toolgrad"
python3 -m pip install -q -e "$DEST/GraphGen" \
    || echo "NOTE: GraphGen editable install failed; the bridge only needs its GraphML output."
echo "done."
