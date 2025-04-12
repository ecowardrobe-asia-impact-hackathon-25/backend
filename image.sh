#!/bin/bash

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

if command -v xdg-open &> /dev/null; then
  OPEN_CMD="xdg-open"
elif command -v open &> /dev/null; then
  OPEN_CMD="open"
fi

for img in test_image/*.{png,jpg}; do
  filename=$(basename "$img")
  extension="${filename##*.}"
  base_name="${filename%.*}"
  output_file="${OUTPUT_DIR}/${base_name}_response.json"
  output_matching="${OUTPUT_DIR}/${base_name}_response_matching.json"

  # First request to /upload endpoint
  curl -X POST -F "file=@${img}" http://localhost:8080/upload > "$output_file"
  
  # Then request to /matching endpoint (using the cached data)

  # if [ -n "$OPEN_CMD" ]; then
  #     "$OPEN_CMD" "$img" &
  # fi
done

echo "All responses have been saved in the '$OUTPUT_DIR' directory."
