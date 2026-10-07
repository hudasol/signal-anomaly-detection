#!/usr/bin/env bash
# Runs the acceptance demo (docs/DEMO.md) step by step, pausing between steps, so the
# video can be recorded in one take. Nothing here edits a model or the official result;
# step 7 renames the model file and ALWAYS puts it back (even on Ctrl-C).
#
#   bash scripts/demo_walkthrough.sh            # pause for Enter between steps
#   NO_PAUSE=1 bash scripts/demo_walkthrough.sh # run straight through (rehearsal)
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m==== %s ====\033[0m\n' "$1"; }
run() { printf '\n$ %s\n' "$*"; "$@"; }
pause() { [ -n "${NO_PAUSE:-}" ] || read -r -p $'\n[Enter] next step ' _; }

ART=$(python -c "import json;r=json.load(open('models/registry.json'));print(r['models'][r['serving']]['artifact'])")
restore() { [ -f "$ART.bak" ] && mv "$ART.bak" "$ART" && echo "model restored: $ART" || true; }
trap restore EXIT

step "1. The split, and proof the test runs were not used"
run signal-data summary
run signal-eval audit
run git log --oneline -1 d121e2a
run git log --oneline -1 eb83aa7
pause

step "2. A normal held-out run: false incidents"
run signal-replay --run r4002 --show-truth
run signal-replay --run r4058 --show-truth
pause

step "3. A held-out fault: score, threshold crossing, incident open/close, latency"
run signal-replay --run r4009 --asset drone-01 --show-truth
run signal-replay --run r4009 --detector rule --asset drone-01 --show-truth
pause

step "4. Rule vs robust z vs LOF vs shipped, same test data"
run signal-eval show
pause

step "5. One false positive, one false negative, one late catch"
run signal-replay --run r4037 --asset quad-01 --show-truth
run signal-replay --run r4023 --asset quad-01 --show-truth
run signal-replay --run r4000 --asset quad-01 --show-truth
pause

step "6. Change the threshold for the demo only, then show it is back"
run signal-eval demo-threshold --detector hybrid_rule_fast --threshold 2.0
run signal-eval demo-threshold --detector hybrid_rule_fast --threshold 0.9
run git status --short results/official models/registry.json
signal-replay --run r4009 --asset drone-01 | head -1
pause

step "7. Remove the model: unavailable, never normal"
mv "$ART" "$ART.bak"
run signal-replay --run r4009 --asset drone-01 || true
restore
pause

step "8. Beyond the bar in one replay: drift check, severity, shadow LOF"
run signal-replay --run r4009 --asset drone-01 --shadow lof --show-truth
pause

step "9. The tests"
run pytest -q tests/test_features.py tests/test_splits.py tests/test_eval.py tests/test_incidents.py tests/test_service.py
run pytest -q
