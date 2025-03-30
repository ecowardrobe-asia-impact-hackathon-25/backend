#!/bin/bash

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

if command -v xdg-open &> /dev/null; then
  OPEN_CMD="xdg-open"
elif command -v open &> /dev/null; then
  OPEN_CMD="open"
fi

for img in test_image/*.png; do
    filename=$(basename "$img" .png)
    output_file="${OUTPUT_DIR}/${filename}_response.json"

    curl -X POST -F "file=@${img}" http://localhost:8080/upload > "$output_file" &

    if [ -n "$OPEN_CMD" ]; then
        "$OPEN_CMD" "$img" &
    fi
done

wait

echo "All responses have been saved in the '$OUTPUT_DIR' directory."
