#!/bin/bash

set -e

cd "/app/eval/evaluation dataset/ai_taiko_set"

for f in *.osu; do
    if python3 ../../osu_to_bin.py "$f"; then
        echo "Successfully converted $f"
    else
        echo "Failed to convert $f" >&2
        exit 1
    fi
done

mkdir -p "$PWD/../../converted_charts/ai_taiko_bin"
mkdir -p "$PWD/../../converted_charts/ai_taiko_time"

for f in *bin.npy; do
    mv -- "$f" "$PWD/../../converted_charts/ai_taiko_bin"
done

for f in *.npy; do
    mv -- "$f" "$PWD/../../converted_charts/ai_taiko_time"
done
