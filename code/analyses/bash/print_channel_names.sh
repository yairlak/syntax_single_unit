#!/bin/bash

for pt in 482 487 489 499 504 510 513 515 530; do
    cmd="python3 ../print_channel_names.py --patient "$pt" --data-type spike"
    echo $cmd
    $cmd
done
