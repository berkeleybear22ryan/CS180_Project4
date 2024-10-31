#!/bin/bash

max_concurrent_jobs=10

output_dir="./video"
mkdir -p "$output_dir"

process_frame() {
    section="$1"
    image_name="$1"

    if [ ! -d "./images/$section" ]; then
        echo "Warning: Directory ./images/$section does not exist. Skipping."
        return
    fi

    echo "Processing section: $section with image name: $image_name"

    echo "starting b1.py ..."
    python b1.py "$section"
    echo "starting b2.py ..."
    python b2.py "$section"
    echo "starting b3.py ..."
    python b3.py "$section"
    echo "starting b4.py ..."
    python b4.py "$section"
    echo "starting b5_1.py ..."
    python b5_1.py "$section"

    src_dir="./part2_output/${section}/points"
    dest_dir="./points/${section}"

    if [ -d "$src_dir" ]; then
        mkdir -p "$dest_dir"
        cp -r "$src_dir/"* "$dest_dir/"
        echo "Files copied from $src_dir to $dest_dir."
    else
        echo "Error: Source directory $src_dir does not exist."
    fi

    python a2.py "$section"
    python a3.py "$section"
    python a5.py "$section"

    mosaic_output="./output/${image_name}/mosaic_sharpened.jpg"
    if [ -f "$mosaic_output" ]; then
        final_image_name=$(printf "%03d.jpg" "$((10#$i - 1))")
        cp "$mosaic_output" "$output_dir/$final_image_name"
        echo "Copied $mosaic_output to $output_dir/$final_image_name"
    else
        echo "Warning: $mosaic_output does not exist for section $section."
    fi
}

for i in $(seq -w 1 700); do
    section="frame_0$i"

    process_frame "$section" &

    while (( $(jobs -r | wc -l) >= max_concurrent_jobs )); do
        sleep 1
    done
done

wait

ffmpeg -framerate 24 -i "$output_dir/%03d.jpg" -c:v libx264 -pix_fmt yuv420p video_output.mp4
echo "Video created as video_output.mp4"
