python encoding_trf.py --level word --patient 479_11 --data-type micro --filter gaussian-kernel --probe-name LSTG --tmin -0.1 --tmax 0.7 --model-type ridge --query "word_length>1" --feature-list letters word_length phone_string is_first_word is_last_word word_position tense pos_simple word_zipf morph_complex grammatical_number embedding wh_subj_obj dec_quest semantic_features --decimate 20
