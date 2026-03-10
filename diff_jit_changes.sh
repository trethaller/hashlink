#!/usr/bin/env bash
set -euo pipefail

diff -u "/d/shiroTools/hashlink/src/src/jit.c" "/d/hashlink/src/jit.c" > "$(dirname "$0")/changes.diff"

echo "Wrote diff to $(dirname "$0")/changes.diff"
