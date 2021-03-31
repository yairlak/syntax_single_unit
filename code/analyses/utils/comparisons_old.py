def comparison_list():
    comparisons = {}

# Sanity checks:
    comparisons[0] = {}
    comparisons[0]['name'] = 'first_last_word_audio'
    comparisons[0]['train_queries'] = ["word_position==1 and first_phone==1 and (block in [2, 4, 6])", "last_word==True and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[0]['train_condition_names'] = ['First_word', 'Last_word']
    comparisons[0]['colors'] = ['b', 'g']

    comparisons[1] = {}
    comparisons[1]['name'] = 'first_last_word_visual'
    comparisons[1]['train_queries'] = ["word_position==1 and (block in [1, 3, 5])", "last_word==True and (block in [1, 3, 5])"]
    comparisons[1]['train_condition_names'] = ['First_word', 'Last_word']
    comparisons[1]['colors'] = ['b', 'g']


# GRAMMATICAL NUMBER:
    # Nouns:
    comparisons[2] = {}
    comparisons[2]['name'] = 'grammatical_number_nouns_audio'
    comparisons[2]['train_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and (block in [2, 4, 6])", "word_position==2 and first_phone==1 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and (block in [2, 4, 6])"]
    comparisons[2]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[2]['colors'] = ['b', 'g']

    comparisons[3] = {}
    comparisons[3]['name'] = 'grammatical_number_nouns_visual'
    comparisons[3]['train_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and (block in [1, 3, 5])", "word_position==2 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and (block in [1, 3, 5])"]
    comparisons[3]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[3]['colors'] = ['b', 'g']

    comparisons[4] = {}
    comparisons[4]['name'] = 'grammatical_number_nouns_audio2visual'
    comparisons[4]['train_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and (block in [2, 4, 6])", "word_position==2 and first_phone==1 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and (block in [2, 4, 6])"]
    comparisons[4]['test_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and (block in [1, 3, 5])", "word_position==2 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and (block in [1, 3, 5])"]
    comparisons[4]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[4]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[4]['colors'] = ['b', 'g']

    # Verbs:
    comparisons[5] = {}
    comparisons[5]['name'] = 'grammatical_number_verbs_audio'
    comparisons[5]['train_queries'] = ["pos=='VBZ' and first_phone==1 and (block in [2, 4, 6])", "pos=='VBP' and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[5]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[5]['colors'] = ['b', 'g']

    comparisons[6] = {}
    comparisons[6]['name'] = 'grammatical_number_verbs_visual'
    comparisons[6]['train_queries'] = ["pos=='VBZ' and (block in [1, 3, 5])", "pos=='VBP' and (block in [1, 3, 5])"]
    comparisons[6]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[6]['colors'] = ['b', 'g']

    comparisons[7] = {}
    comparisons[7]['name'] = 'grammatical_number_verbs_audio2visual'
    comparisons[7]['train_queries'] = ["pos=='VBZ' and first_phone==1 and (block in [2, 4, 6])", "pos=='VBP' and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[7]['test_queries'] = ["pos=='VBZ' and (block in [1, 3, 5])", "pos=='VBP' and (block in [1, 3, 5])"]
    comparisons[7]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[7]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[7]['colors'] = ['b', 'g']

    # Pronouns:
    comparisons[8] = {}
    comparisons[8]['name'] = 'grammatical_number_pronouns_audio'
    comparisons[8]['train_queries'] = ["word_position==1 and first_phone==1 and (word_string=='he' or word_string=='she') and (block in [2, 4, 6])", "word_position==1 and first_phone==1 and (word_string=='they') and (block in [2, 4, 6])"]
    comparisons[8]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[8]['colors'] = ['b', 'g']

    comparisons[9] = {}
    comparisons[9]['name'] = 'grammatical_number_pronouns_visual'
    comparisons[9]['train_queries'] = ["word_position==1 and (word_string=='he' or word_string=='she') and (block in [1, 3, 5])", "word_position==1 and (word_string=='they') and (block in [1, 3, 5])"]
    comparisons[9]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[9]['colors'] = ['b', 'g']

    comparisons[10] = {}
    comparisons[10]['name'] = 'grammatical_number_pronouns_audio2visual'
    comparisons[10]['train_queries'] = ["word_position==1 and first_phone==1 and (word_string=='he' or word_string=='she') and (block in [2, 4, 6])", "word_position==1 and first_phone==1 and (word_string=='they') and (block in [2, 4, 6])"]
    comparisons[10]['test_queries'] = ["word_position==1 and (word_string=='he' or word_string=='she') and (block in [1, 3, 5])", "word_position==1 and (word_string=='they') and (block in [1, 3, 5])"]
    comparisons[10]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[10]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[10]['colors'] = ['b', 'g']

    #Nouns2Verbs-audio
    comparisons[11] = {}
    comparisons[11]['name'] = 'grammatical_number_nouns2verbs_audio'
    comparisons[11]['train_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and (block in [2, 4, 6])", "word_position==2 and first_phone==1 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and (block in [2, 4, 6])"]
    comparisons[11]['test_queries'] = ["pos=='VBZ' and first_phone==1 and (block in [2, 4, 6])", "pos=='VBP' and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[11]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[11]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[11]['colors'] = ['b', 'g']

    #Nouns2Verbs-visual
    comparisons[12] = {}
    comparisons[12]['name'] = 'grammatical_number_nouns2verbs_visual'
    comparisons[12]['train_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and (block in [1, 3, 5])", "word_position==2 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and (block in [1, 3, 5])"]
    comparisons[12]['test_queries'] = ["pos=='VBZ' and (block in [1, 3, 5])", "pos=='VBP' and (block in [1, 3, 5])"]
    comparisons[12]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[12]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[12]['colors'] = ['b', 'g']

    #Nouns2Pronouns-audio
    comparisons[13] = {}
    comparisons[13]['name'] = 'grammatical_number_nouns2pronouns_audio'
    comparisons[13]['train_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and block in [2, 4, 6]", "word_position==2 and first_phone==1 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and block in [2, 4, 6]"]
    comparisons[13]['test_queries'] = ["word_position==1 and first_phone==1 and (word_string=='he' or word_string=='she') and block in [2, 4, 6]", "word_position==1 and (word_string=='they') and block in [2, 4, 6]"]
    comparisons[13]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[13]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[13]['colors'] = ['b', 'g']

    #Nouns2Pronouns-visual
    comparisons[14] = {}
    comparisons[14]['name'] = 'grammatical_number_nouns2pronouns_visual'
    comparisons[14]['train_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='girl' or word_string=='man' or word_string=='woman') and block in [1, 3, 5]", "word_position==2 and (word_string=='boys' or word_string=='girls' or word_string=='men' or word_string=='women') and block in [1, 3, 5]"]
    comparisons[14]['test_queries'] = ["word_position==1 and (word_string=='he' or word_string=='she') and block in [1, 3, 5]", "word_position==1 and (word_string=='they') and block in [1, 3, 5]"]
    comparisons[14]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[14]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[14]['colors'] = ['b', 'g']

    #Pronouns2Verbs-audio
    comparisons[15] = {}
    comparisons[15]['name'] = 'grammatical_number_pronouns2verbs_audio'
    comparisons[15]['train_queries'] = ["word_position==1 and first_phone==1 and (word_string=='he' or word_string=='she') and block in [2, 4, 6]", "word_position==1 and first_phone==1 and (word_string=='they') and block in [2, 4, 6]"]
    comparisons[15]['test_queries'] = ["pos=='VBZ' and first_phone==1 and block in [2, 4, 6]", "pos=='VBP' and first_phone==1 and block in [2, 4, 6]"]
    comparisons[15]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[15]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[15]['colors'] = ['b', 'g']

    #Pronouns2Verbs-visual
    comparisons[16] = {}
    comparisons[16]['name'] = 'grammatical_number_pronouns2verbs_visual'
    comparisons[16]['train_queries'] = ["word_position==1 and (word_string=='he' or word_string=='she') and block in [1, 3, 5]", "word_position==1 and (word_string=='they') and block in [1, 3, 5]"]
    comparisons[16]['test_queries'] = ["pos=='VBZ' and block in [1, 3, 5]", "pos=='VBP' and block in [1, 3, 5]"]
    comparisons[16]['train_condition_names'] = ['Singular', 'Plural']
    comparisons[16]['test_condition_names'] = ['Singular', 'Plural']
    comparisons[16]['colors'] = ['b', 'g']

# Nouns vs. verbs

    comparisons[17] = {}
    comparisons[17]['name'] = 'nouns_verbs_audio'
    comparisons[17]['train_queries'] = ["(pos=='NN' or pos=='NNS') and first_phone==1 and (block in [2, 4, 6])", "(pos=='VBZ' or pos=='VBP') and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[17]['train_condition_names'] = ['Noun', 'Verb']
    comparisons[17]['colors'] = ['b', 'g']

    comparisons[18] = {}
    comparisons[18]['name'] = 'nouns_verbs_visual'
    comparisons[18]['train_queries'] = ["(pos=='NN' or pos=='NNS') and block in [1, 3, 5]", "(pos=='VBZ' or pos=='VBP') and block in [1, 3, 5]"]
    comparisons[18]['train_condition_names'] = ['Noun', 'Verb']
    comparisons[18]['colors'] = ['b', 'g']

    comparisons[19] = {}
    comparisons[19]['name'] = 'nouns_verbs_audio2visual'
    comparisons[19]['train_queries'] = ["(pos=='NN' or pos=='NNS') and first_phone==1 and (block in [2, 4, 6])", "(pos=='VBZ' or pos=='VBP') and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[19]['test_queries'] = ["(pos=='NN' or pos=='NNS') and block in [1, 3, 5]", "(pos=='VBZ' or pos=='VBP') and block in [1, 3, 5]"]
    comparisons[19]['train_condition_names'] = ['Noun', 'Verb']
    comparisons[19]['test_condition_names'] = ['Noun', 'Verb']
    comparisons[19]['colors'] = ['b', 'g']

# Gender
    
    comparisons[20] = {}
    comparisons[20]['name'] = 'gender_pronouns_audio'
    comparisons[20]['train_queries'] = ["word_position==1 and first_phone==1 and word_string=='he' and (block in [2, 4, 6])", "word_position==1 and first_phone==1 and word_string=='she' and (block in [2, 4, 6])"]
    comparisons[20]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[20]['colors'] = ['b', 'g']

    comparisons[21] = {}
    comparisons[21]['name'] = 'gender_pronouns_visual'
    comparisons[21]['train_queries'] = ["word_position==1 and word_string=='he' and (block in [1, 3, 5])", "word_position==1 and word_string=='she' and (block in [1, 3, 5])"]
    comparisons[21]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[21]['colors'] = ['b', 'g']


    comparisons[22] = {}
    comparisons[22]['name'] = 'gender_pronouns_audio2visual'
    comparisons[22]['train_queries'] = ["word_position==1 and first_phone==1 and word_string=='he' and (block in [2, 4, 6])", "word_position==1 and first_phone==1 and word_string=='she' and (block in [2, 4, 6])"]
    comparisons[22]['test_queries'] = ["word_position==1 and word_string=='he' and block in [1, 3, 5]", "word_position==1 and word_string=='she' and block in [1, 3, 5]"]
    comparisons[22]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[22]['test_condition_names'] = ['Masculine', 'Feminine']
    comparisons[22]['colors'] = ['b', 'g']
    
    comparisons[23] = {}
    comparisons[23]['name'] = 'gender_nouns_audio'
    comparisons[23]['train_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men') and block in [2, 4, 6]", "word_position==2 and first_phone==1 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women') and block in [2, 4, 6]"]
    comparisons[23]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[23]['colors'] = ['b', 'g']

    comparisons[24] = {}
    comparisons[24]['name'] = 'gender_nouns_visual'
    comparisons[24]['train_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men') and block in [1, 3, 5]", "word_position==2 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women') and block in [1, 3, 5]"]
    comparisons[24]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[24]['colors'] = ['b', 'g']

    comparisons[25] = {}
    comparisons[25]['name'] = 'gender_nouns_audio2visual'
    comparisons[25]['train_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men') and block in [2, 4, 6]", "word_position==2 and first_phone==1 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women') and block in [2, 4, 6]"]
    comparisons[25]['test_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men') and block in [1, 3, 5]", "word_position==2 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women') and block in [1, 3, 5]"]
    comparisons[25]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[25]['test_condition_names'] = ['Masculine', 'Feminine']
    comparisons[25]['colors'] = ['b', 'g']
    
    comparisons[26] = {}
    comparisons[26]['name'] = 'gender_pronoun2noun_audio'
    comparisons[26]['train_queries'] = ["word_position==1 and first_phone==1 and word_string=='he' and block in [2, 4, 6]", "word_position==1 and first_phone==1 and word_string=='she' and block in [2, 4, 6]"]
    comparisons[26]['test_queries'] = ["word_position==2 and first_phone==1 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men') and block in [2, 4, 6]", "word_position==2 and first_phone==1 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women') and block in [2, 4, 6]"]
    comparisons[26]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[26]['test_condition_names'] = ['Masculine', 'Feminine']
    comparisons[26]['colors'] = ['b', 'g']

    comparisons[27] = {}
    comparisons[27]['name'] = 'gender_pronoun2noun_visual'
    comparisons[27]['train_queries'] = ["word_position==1 and word_string=='he' and block in [1, 3, 5]", "word_position==1 and word_string=='she' and block in [1, 3, 5]"]
    comparisons[27]['test_queries'] = ["word_position==2 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men') and block in [1, 3, 5]", "word_position==2 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women') and block in [1, 3, 5]"]
    comparisons[27]['train_condition_names'] = ['Masculine', 'Feminine']
    comparisons[27]['test_condition_names'] = ['Masculine', 'Feminine']
    comparisons[27]['colors'] = ['b', 'g']

# Sentence type
    
    comparisons[28] = {}
    comparisons[28]['name'] = 'declarative_questions_auditory'
    comparisons[28]['train_queries'] = ["Declarative==1 and (block in [2, 4, 6]) and last_word==True and first_phone==1", "Question==1 and (block in [2, 4, 6]) and last_word==True and first_phone==1"]
    comparisons[28]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[28]['colors'] = ['b', 'g']

    comparisons[29] = {}
    comparisons[29]['name'] = 'declarative_questions_visual'
    comparisons[29]['train_queries'] = ["Declarative==1 and block in [1, 3, 5] and last_word==True", "Question==1 and block in [1, 3, 5] and last_word==True"]
    comparisons[29]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[29]['colors'] = ['b', 'g']

    comparisons[30] = {}
    comparisons[30]['name'] = 'declarative_questions_auditory2visual'
    comparisons[30]['train_queries'] = ["Declarative==1 and block in [2, 4, 6] and last_word==True and first_phone==1", "Question==1 and block in [2, 4, 6] and last_word==True and first_phone==1"]
    comparisons[30]['test_queries'] = ["Declarative==1 and block in [1, 3, 5] and last_word==True", "Question==1 and block in [1, 3, 5] and last_word==True"]
    comparisons[30]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[30]['test_condition_names'] = ['Declarative', 'Question']
    comparisons[30]['colors'] = ['b', 'g']
    
# Embedding

    comparisons[31] = {}
    comparisons[31]['name'] = 'embedding_auditory'
    comparisons[31]['train_queries'] = ["Embedding==1 and last_word==True and first_phone==1 and block in [2, 4, 6]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and first_phone==1 and block in [2, 4, 6]"]
    comparisons[31]['train_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[31]['colors'] = ['b', 'g']

    comparisons[32] = {}
    comparisons[32]['name'] = 'embedding_visual'
    comparisons[32]['train_queries'] = ["Embedding==1 and last_word==True and block in [1, 3, 5]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and block in [1, 3, 5]"]
    comparisons[32]['train_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[32]['colors'] = ['b', 'g']

    comparisons[33] = {}
    comparisons[33]['name'] = 'embedding_audio2visual'
    comparisons[33]['train_queries'] = ["Embedding==1 and last_word==True and first_phone==1 and block in [2, 4, 6]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and first_phone==1 and block in [2, 4, 6]"]
    comparisons[33]['test_queries'] = ["Embedding == 1 and last_word==True and block in [1, 3, 5]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and block in [1, 3, 5]"]
    comparisons[33]['train_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[33]['test_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[33]['colors'] = ['b', 'g']

# Number of letters
    comparisons[34] = {}
    comparisons[34]['name'] = 'word_length_audio'
    comparisons[34]['train_queries'] = ["word_string.str.len()<4 and word_string.str.len()>1 and block in [2, 4, 6] and first_phone==1", "word_string.str.len()>7 and block in [2, 4, 6] and first_phone==1"]
    comparisons[34]['train_condition_names'] = ['Short_word', 'Long_word']
    comparisons[34]['colors'] = ['b', 'g']

    comparisons[35] = {}
    comparisons[35]['name'] = 'word_length_visual'
    comparisons[35]['train_queries'] = ["word_string.str.len()<4 and word_string.str.len()>1 and block in [1, 3, 5]", "word_string.str.len()>7 and block in [1, 3, 5]"]
    comparisons[35]['train_condition_names'] = ['Short_word', 'Long_word']
    comparisons[35]['colors'] = ['b', 'g']
   
# Phonemes
    vowels, consonants = get_phone_classes()
    
    # PLACE
    comparisons[36] = {}
    comparisons[36]['name'] = 'place_of_articulation'
    comparisons[36]['train_queries'] = []
    comparisons[36]['train_condition_names'] = []
    for cl in ['COR', 'LAB', 'DOR']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[36]['train_queries'].append(curr_query)
        comparisons[36]['train_condition_names'].append(cl)
    comparisons[36]['colors'] = ['r', 'g', 'b']

    # MANNER
    comparisons[37] = {}
    comparisons[37]['name'] = 'manner_of_articulation'
    comparisons[37]['train_queries'] = []
    comparisons[37]['train_condition_names'] = []
    for cl in ['STOP', 'AFFR', 'FRIC', 'NAS', 'LIQ', 'SIB', 'GLI']:
        curr_class = consonants[cl].split(' ')
        #curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_class = ['"'+s.strip(' ')+'"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[37]['train_queries'].append(curr_query)
        comparisons[37]['train_condition_names'].append(cl)
    comparisons[37]['colors'] = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    # SONORITY
    comparisons[38] = {}
    comparisons[38]['name'] = 'sonority'
    comparisons[38]['train_queries'] = []
    comparisons[38]['train_condition_names'] = []
    for cl in ['VLS', 'VOI', 'SON']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[38]['train_queries'].append(curr_query)
        comparisons[38]['train_condition_names'].append(cl)
    comparisons[38]['colors'] = ['r', 'g', 'b']
    
    # BACKNESS and HEIGHT 
    comparisons[39] = {}
    comparisons[39]['name'] = 'backness_height'
    comparisons[39]['train_queries'] = []
    comparisons[39]['train_condition_names'] = []
    for cl in ['LOW', 'MID', 'HI', 'BCK', 'CNT', 'FRO', 'DIPH']:
        curr_class = vowels[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[39]['train_queries'].append(curr_query)
        comparisons[39]['train_condition_names'].append(cl)
    comparisons[39]['colors'] = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    # ROUNDEDNESS
    comparisons[40] = {}
    comparisons[40]['name'] = 'roundedness'
    comparisons[40]['train_queries'] = []
    comparisons[40]['train_condition_names'] = []
    for cl in ['RND', 'UNR']:
        curr_class = vowels[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[40]['train_queries'].append(curr_query)
        comparisons[40]['train_condition_names'].append(cl)
    comparisons[40]['colors'] = ['r', 'g']

    # TENSENESS
    comparisons[41] = {}
    comparisons[41]['name'] = 'tenseness'
    comparisons[41]['train_queries'] = []
    comparisons[41]['train_condition_names'] = []
    for cl in ['TNS', 'LAX']:
        curr_class = vowels[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[41]['train_queries'].append(curr_query)
        comparisons[41]['train_condition_names'].append(cl)
    comparisons[41]['colors'] = ['r', 'g']

# WORD POSITION
    comparisons[42] = {}
    comparisons[42]['name'] = 'word_position_audio'
    comparisons[42]['train_queries'] = ["word_position==%i and first_phone==1 and (block in [2, 4, 6])"%i for i in range(1, 6)]
    comparisons[42]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[42]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[43] = {}
    comparisons[43]['name'] = 'word_position_visual'
    comparisons[43]['train_queries'] = ["word_position==%i and (block in [1, 3, 5])"%i for i in range(1, 6)]
    comparisons[43]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[43]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[44] = {}
    comparisons[44]['name'] = 'word_position_audio2visual'
    comparisons[44]['train_queries'] = ["word_position==%i and first_phone==1 and (block in [2, 4, 6])"%i for i in range(1, 6)]
    comparisons[44]['test_queries'] = ["word_position==%i and (block in [1, 3, 5])"%i for i in range(1, 6)]
    comparisons[44]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[44]['test_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[44]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[45] = {}
    comparisons[45]['name'] = 'num_words_audio'
    comparisons[45]['train_queries'] = ["word_position==%i and first_phone==1 and last_word==True and (block in [2, 4, 6])"%i for i in range(1, 6)]
    comparisons[45]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[45]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[46] = {}
    comparisons[46]['name'] = 'word_position_visual'
    comparisons[46]['train_queries'] = ["word_position==%i and last_word==True and (block in [1, 3, 5])"%i for i in range(1, 6)]
    comparisons[46]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[46]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[47] = {}
    comparisons[47]['name'] = 'word_position_audio2visual'
    comparisons[47]['train_queries'] = ["word_position==%i and first_phone==1 and last_word==True and (block in [2, 4, 6])"%i for i in range(1, 6)]
    comparisons[47]['test_queries'] = ["word_position==%i and last_word==True and (block in [1, 3, 5])"%i for i in range(1, 6)]
    comparisons[47]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[47]['test_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[47]['colors'] = ['r', 'g', 'b', 'c', 'm']

# WORD FREQ
    comparisons[48] = {}
    comparisons[48]['name'] = 'word_zipf_audio'
    comparisons[48]['train_queries'] = ["word_freq < 0.0001", "word_zipf > 6" ]
    #comparisons[48]['train_queries'] = ["word_zipf>%i and word_zipf<=%i and first_phone==1 and (block in [2, 4, 6])"%(i, j) for (i, j) in zip(range(0, 8), range(1, 9))]
    comparisons[48]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[48]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[49] = {}
    comparisons[49]['name'] = 'word_zipf_visual'
    comparisons[49]['train_queries'] = ["(word_zipf>%i) and (word_zipf<=%i) and (block in [1, 3, 5])"%(i, j) for (i, j) in zip(range(0, 8), range(1, 9))]
    comparisons[49]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[49]['colors'] = ['r', 'g', 'b', 'c', 'm']

    comparisons[50] = {}
    comparisons[50]['name'] = 'word_zipf_audio2visual'
    comparisons[50]['train_queries'] = ["word_zipf>%i and word_zipf<=%i and first_phone==1 and (block in [2, 4, 6])"%(i, j) for (i, j) in zip(range(0, 8), range(1, 9))]
    comparisons[50]['test_queries'] = ["word_zipf>%i and word_zipf<=%i and (block in [1, 3, 5])"%(i, j) for (i, j) in zip(range(0, 8), range(1, 9))]
    comparisons[50]['train_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[50]['test_condition_names'] = ['1', '2', '3', '4', '5']
    comparisons[50]['colors'] = ['r', 'g', 'b', 'c', 'm']
    
    
    comparisons[51] = {}
    comparisons[51]['name'] = 'declarative_questions_auditory_onset'
    comparisons[51]['train_queries'] = ["Declarative==1 and (block in [2, 4, 6]) and word_position==1", "Question==1 and (block in [2, 4, 6]) and word_position==1"]
    comparisons[51]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[51]['colors'] = ['b', 'g']
    
    return comparisons


############################################################################################################
############################################################################################################
############################################################################################################

def get_phone_classes():
    VOWELS = {}
    CONSONANTS = {}
    ###VOWEL QUALITY
    VOWELS['TNS'] = "IY1 IY2 IY0 EY1 EY2 EY0 EYR1 EYR2 EYR0 OW1 OW2 OW0 UW1 UW2 UW0"
    VOWELS['LAX'] = "IH1 IH2 IH0 IR1 IR2 IR0 EH1 EH2 EH0 AH1 AH2 AH0 AE1 AE2 AE0 AY1 AY2 AY0 AW1 AW2 AW0 AA1 AA2 AA0 AAR1 AAR2 AAR0 AO1 AO2 AO0 OY1 OY2 OY0 OR1 OR2 OR0 ER1 ER2 ER0 UH1 UH2 UH0 UR1 UR2 UR0"
    VOWELS['UNR'] = "IY1 IY2 IY0 IH1 IH2 IH0 IR1 IR2 IR0 EY1 EY2 EY0 EYR1 EYR2 EYR0 EH1 EH2 EH0 AH1 AH2 AH0 AE1 AE2 AE0 AY1 AY2 AY0 AA1 AA2 AA0 AAR1 AAR2 AAR0"
    VOWELS['RND'] = "AW1 AW2 AW0 AO1 AO2 AO0 OW1 OW2 OW0 OY1 OY2 OY0 OR1 OR2 OR0 ER1 ER2 ER0 UH1 UH2 UH0 UW1 UW2 UW0 UR1 UR2 UR0"
    VOWELS['BCK'] = "AA1 AA2 AA0 AAR1 AAR2 AAR0 AO1 AO2 AO0 OW1 OW2 OW0 OY1 OY2 OY0 OR1 OR2 OR0 UH1 UH2 UH0 UW1 UW2 UW0 UR1 UR2 UR0"
    VOWELS['CNT'] = "AH1 AH2 AH0 AY1 AY2 AY0 AW1 AW2 AW0 ER1 ER2 ER0"
    VOWELS['FRO'] = "IY1 IY2 IY0 IH1 IH2 IH0 IR1 IR2 IR0 EY1 EY2 EY0 EYR1 EYR2 EYR0 EH1 EH2 EH0 AE1 AE2 AE0"
    VOWELS['HI'] = "IY1 IY2 IY0 IH1 IH2 IH0 IR1 IR2 IR0 UH1 UH2 UH0 UW1 UW2 UW0 UR1 UR2 UR0"
    VOWELS['MID'] = "EY1 EY2 EY0 EYR1 EYR2 EYR0 EH1 EH2 EH0 AH1 AH2 AH0 AO1 AO2 AO0 OW1 OW2 OW0 OY1 OY2 OY0 OR1 OR2 OR0 ER1 ER2 ER0"
    VOWELS['LOW'] = "AE1 AE2 AE0 AY1 AY2 AY0 AW1 AW2 AW0 AA1 AA2 AA0 AAR1 AAR2 AAR0"
    VOWELS['DIPH'] = "AY1 AY2 AY0 AW1 AW2 AW0 OY1 OY2 OY0"

    ###VOWEL STRESS
    VOWELS['STR'] = "IY1 IH1 IR1 EY1 EYR1 EH1 AH1 AE1 AY1 AW1 AA1 AAR1 AO1 OW1 OY1 OR1 ER1 UH1 UW1 UR1"
    VOWELS['2ND'] = "IY2 IH2 IR2 EY2 EYR2 EH2 AH2 AE2 AY2 AW2 AA2 AAR2 AO2 OW2 OY2 OR2 ER2 UH2 UW2 UR2"
    VOWELS['UNS'] = "IY0 IH0 IR0 EY0 EYR0 EH0 AH0 AE0 AY0 AW0 AA0 AAR0 AO0 OW0 OY0 OR0 ER0 UH0 UW0 UR0"

    ###CONSONANT PLACE
    #CONSONANTS['COR'] = "DENT ALV POSTALV PAL LIQ"
    #CONSONANTS['LAB'] = "BILAB LABDENT W"
    #CONSONANTS['DOR'] = "VEL W"
    CONSONANTS['COR'] = "TH DH T D S Z N CH JH SH ZH Y L R"
    CONSONANTS['LAB'] = "P B M F V W"
    CONSONANTS['DOR'] = "K G NG W"
    CONSONANTS['BILAB'] = "P B M"
    CONSONANTS['LABDENT'] = "F V"
    CONSONANTS['DENT'] = "TH DH"
    CONSONANTS['ALV'] = "T D S Z N"
    CONSONANTS['PLV'] = "CH JH SH ZH"
    CONSONANTS['PAL'] = "Y"
    CONSONANTS['VEL'] = "K G NG"
    CONSONANTS['LAR'] = "HH"

    ###CONSONANT MANNER
    CONSONANTS['STOP'] = "P T K B D G"
    CONSONANTS['AFFR'] = "CH JH"
    CONSONANTS['FRIC'] = "F TH S SH HH V DH Z ZH"
    CONSONANTS['NAS'] = "M N NG"
    CONSONANTS['LIQ'] = "L R"
    CONSONANTS['SIB'] = "CH S SH JH Z ZH"
    CONSONANTS['GLI'] = "W Y"

    ###CONSONANT VOICING/SONORANCE
    CONSONANTS['VLS'] = "P T K CH F TH S SH HH"
    CONSONANTS['VOI'] = "B D G JH V DH Z ZH"
    CONSONANTS['SON'] = "M N NG L W R Y"

    return VOWELS, CONSONANTS
