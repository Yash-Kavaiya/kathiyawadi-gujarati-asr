#!/usr/bin/env bash
# Build a word n-gram LM for CTC beam-search decoding.
#   bash lm/build_kenlm.sh corpus1.txt [corpus2.txt ...]
# Corpora: IndicCorp v2 gu, Sangraha gu, plus training transcripts.
# NEVER include dev/test transcripts. Needs KenLM binaries (lmplz, build_binary) on PATH.
set -euo pipefail

ORDER=${ORDER:-5}
OUT=${OUT:-lm/gu_${ORDER}gram}
mkdir -p "$(dirname "$OUT")"

# Same normaliser as training targets, one sentence per line, deduplicated.
python - "$@" > "$OUT.txt" <<'EOF'
import sys
from kgasr.normalize import normalize, is_mostly_gujarati
seen = set()
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8", errors="ignore"):
        t = normalize(line, remove_nukta=False, candrabindu_to_anusvara=False)
        if t and is_mostly_gujarati(t) and t not in seen:
            seen.add(t)
            print(t)
EOF

lmplz -o "$ORDER" --prune 0 0 1 --text "$OUT.txt" --arpa "$OUT.arpa"
build_binary "$OUT.arpa" "$OUT.bin"
echo "LM: $OUT.bin ($(wc -l < "$OUT.txt") sentences)"
