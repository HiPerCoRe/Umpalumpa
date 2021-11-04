#!/usr/bin/env bash
set -e
find $(dirname "$0") -type f -executable -not -path "$0" -print0 | while IFS= read -r -d '' test; do
    "$test"
done
