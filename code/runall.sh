#!/bin/bash

section="T_P2"
echo "Section is set to: $section"

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
    echo "Files copied from $src_dir."
    echo "Files copied to $dest_dir."
else
    echo "Error: Source directory $src_dir does not exist."
fi

python a2.py "$section"
python a3.py "$section"
python a5.py "$section"