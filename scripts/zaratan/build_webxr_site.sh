#!/bin/bash
# build_webxr_site.sh — collect outputs/splats/*.ply into a static WebXR site.
#
# Output layout under $OUT_ROOT/www:
#   www/
#   ├── index.html           (redirect to viewer/)
#   ├── viewer/              (copy of project's viewer/)
#   │   ├── index.html
#   │   └── scenes.json      (regenerated, points at relative ../splats/*.ply)
#   └── splats/              (symlinks to the actual .ply files)
#
# The viewer is mkkellogg/gaussian-splats-3d via CDN — static HTTP serve works.
# .ksplat conversion is left for a separate step (requires npm).

set -uo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch/phase4/textworld-vr}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT}"
CODE_ROOT="${CODE_ROOT:-$SCRATCH_ROOT/code}"
SPLATS_SRC="${SPLATS_SRC:-$SCRATCH_ROOT/outputs/splats}"
# Layered 3DGS (LayerPano3D) outputs — add to the site alongside v2 splats
SPLATS_LAYERED="${SPLATS_LAYERED:-$SCRATCH_ROOT/outputs_v3/splats}"
# Also pull from BeeOND if batch is still writing there
BEEOND_SPLATS="${BEEOND_SPLATS:-}"
WWW="$OUT_ROOT/www"
VIEWER_DST="$WWW/viewer"
SPLATS_DST="$WWW/splats"

echo "== building WebXR static site"
echo "   SCRATCH_ROOT=$SCRATCH_ROOT"
echo "   SPLATS_SRC=$SPLATS_SRC"
echo "   BEEOND_SPLATS=$BEEOND_SPLATS"
echo "   WWW=$WWW"

mkdir -p "$WWW" "$VIEWER_DST" "$SPLATS_DST"

# Clean stale splat symlinks/files from prior runs so scenes.json stays in sync with SPLATS_SRC
find "$SPLATS_DST" -maxdepth 1 \( -type l -o -type f \) -name '*.ply' -delete 2>/dev/null || true

# 1. Copy viewer
if [ -d "$CODE_ROOT/viewer" ]; then
    cp -r "$CODE_ROOT/viewer"/* "$VIEWER_DST/"
    echo "   viewer copied from $CODE_ROOT/viewer/"
else
    echo "ERROR: viewer source not found at $CODE_ROOT/viewer" >&2
    exit 1
fi

# 2. Collect splats (both from durable scratch and BeeOND)
echo "== collecting splats"
shopt -s nullglob
splats=()
for f in "$SPLATS_SRC"/*.ply; do splats+=("$f"); done
if [ -d "$SPLATS_LAYERED" ]; then
    for f in "$SPLATS_LAYERED"/*.ply; do splats+=("$f"); done
fi
if [ -n "$BEEOND_SPLATS" ] && [ -d "$BEEOND_SPLATS" ]; then
    for f in "$BEEOND_SPLATS"/*.ply; do
        # Skip if already have an identical one in SPLATS_SRC
        base="$(basename "$f")"
        if [ ! -e "$SPLATS_SRC/$base" ]; then
            splats+=("$f")
        fi
    done
fi

n=${#splats[@]}
echo "   found $n splat files"

# 3. Symlink into www/splats/
for f in "${splats[@]}"; do
    ln -sf "$f" "$SPLATS_DST/$(basename "$f")"
done

# 4. Regenerate scenes.json
export WWW
python - <<PY
import json, os, re
from pathlib import Path

www = Path(os.environ["WWW"])
splat_dir = www / "splats"
scenes = []
# Scene labels: strip "_gsplat" suffix, replace underscores
for p in sorted(splat_dir.glob("*.ply")):
    stem = p.stem
    pipeline = "depth-gsplat"
    scene_id = stem
    if stem.endswith("_layered"):
        pipeline = "layered"
        scene_id = stem[: -len("_layered")]
    elif stem.endswith("_gsplat"):
        pipeline = "depth-gsplat"
        scene_id = stem[: -len("_gsplat")]
    m = re.match(r"^(.*)_s(\d+)$", scene_id)
    if m:
        pretty = m.group(1).replace("_", " ").strip()
        seed = int(m.group(2))
    else:
        pretty = scene_id.replace("_", " ").strip()
        seed = None
    tag = "[layered]" if pipeline == "layered" else "[v2]"
    name = f"{tag} {pretty[:56]}" + (f" (seed {seed})" if seed is not None else "")
    url = f"../splats/{p.name}"
    scenes.append({
        "id": stem,
        "name": name,
        "url": url,
        "pipeline": pipeline,
        # keep legacy aliases so older viewers still work
        "title": name,
        "splat": url,
    })
(www / "viewer" / "scenes.json").write_text(json.dumps(scenes, indent=2))
print(f"wrote {len(scenes)} scenes to viewer/scenes.json")
PY

# 5. Landing-page redirect
cat > "$WWW/index.html" <<'HTML'
<!DOCTYPE html>
<meta charset="utf-8" />
<meta http-equiv="refresh" content="0;url=viewer/" />
<title>TextWorld VR</title>
<p>Redirecting to <a href="viewer/">viewer</a>&hellip;</p>
HTML

# 6. Summary
echo
echo "== site ready"
du -sh "$WWW" 2>/dev/null
echo "   splats: $n"
echo "   viewer: $VIEWER_DST/"
echo "   scenes.json: $VIEWER_DST/scenes.json"
echo
echo "Serve with: python -m http.server 8765 --directory $WWW"
