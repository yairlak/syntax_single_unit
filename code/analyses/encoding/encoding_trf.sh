

#python encoding_trf.py --decimate 20 --patient 479_11 --query "block in [1,3,5]" --data-type micro --filter gaussian-kernel --feature-list is_first_word is_last_word word_zipf letters grammatical_number dec_quest word_length word_position tense pos_simple morph_complex embedding wh_subj_obj semantic_features
#python plot_encoding_trf.py --patient 479_11 --query "block in [1,3,5]" --data-type micro --filter gaussian-kernel

python encoding_trf.py --decimate 20 --patient 479_11 --query "block in [2,4,6]" --data-type micro --filter gaussian-kernel --feature-list is_first_word is_last_word word_zipf phonological_features grammatical_number dec_quest word_length word_position tense pos_simple morph_complex embedding wh_subj_obj semantic_features
python plot_encoding_trf.py --patient 479_11 --query "block in [2,4,6]" --data-type micro --filter gaussian-kernel

python encoding_trf.py --decimate 20 --patient 502 --query "block in [1,3,5]" --data-type micro --filter gaussian-kernel --feature-list is_first_word is_last_word word_zipf letters grammatical_number dec_quest word_length word_position tense pos_simple morph_complex embedding wh_subj_obj semantic_features
python plot_encoding_trf.py --patient 502 --query "block in [1,3,5]" --data-type micro --filter gaussian-kernel

python encoding_trf.py --decimate 20 --patient 502 --query "block in [2,4,6]" --data-type micro --filter gaussian-kernel --feature-list is_first_word is_last_word word_zipf phonological_features grammatical_number dec_quest word_length word_position tense pos_simple morph_complex embedding wh_subj_obj semantic_features
python plot_encoding_trf.py --patient 502 --query "block in [2,4,6]" --data-type micro --filter gaussian-kernel
