python plot_encoding.py --level word --patient 502 --data-type micro --filter gaussian-kernel --probe-name RFSG --model-type ridge --query "word_length>1 and (block in [1, 3, 5])"
python plot_encoding.py --level word --patient 479_11 --data-type micro --filter gaussian-kernel --probe-name LSTG --model-type ridge --query "word_length>1 and (block in [2, 4, 6])"
