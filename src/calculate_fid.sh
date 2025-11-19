#!/bin/bash

# Wait for generation to complete
while true; do
    count=$(find "/DATA1/yunzhu/image generation/generated_images" -name "*.png" 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated images: $count / 10000"

    if [ "$count" -ge 10000 ]; then
        echo "Generation complete! Starting FID calculation..."
        break
    fi

    sleep 60
done

# Calculate FID score
echo "Calculating FID score..."
python -m pytorch_fid "/DATA1/yunzhu/image generation/generated_images" "/DATA1/yunzhu/image generation"

echo "FID calculation complete!"
