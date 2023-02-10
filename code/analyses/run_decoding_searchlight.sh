block_trains='auditory visual'
block_tests='auditory visual'
for block_train in $block_trains; do
	for block_test in $block_tests; do
              cmd='nohup python3 run_decoding_searchlight.py --block-train '$block_train' --block-test '$block_test' --cluster --launch &'
	      eval $cmd
	done
done
