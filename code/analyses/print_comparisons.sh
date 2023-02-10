
comparisons='dec_quest_len2 embedding_vs_long_end embedding_vs_long_3rd_word number_subject number_verb unacc_unerg_dec'
for comparison in $comparisons; do
    python print_comparisons.py --comparison $comparison
done
