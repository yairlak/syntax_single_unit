def comparison_list():
    comparisons = {}

# ALL TRIALS
    comparisons['all_end_trials'] = {}
    comparisons['all_end_trials']['queries'] = ["word_position==-1 and (block in [1, 3, 5])", "word_position==-1 and (block in [2, 4, 6])"]
    comparisons['all_end_trials']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_end_trials']['colors'] = ['b', 'r']
    comparisons['all_end_trials']['sort'] = ['sentence_length', 'sentence_string']

# ALL TRIALS
    comparisons['all_trials'] = {}
    comparisons['all_trials']['queries'] = ["word_position==1 and (block in [1, 3, 5])", "word_position==1 and (block in [2, 4, 6])"]
    comparisons['all_trials']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_trials']['colors'] = ['b', 'r']
    comparisons['all_trials']['sort'] = ['sentence_length', 'sentence_string']#, 'Question']
    comparisons['all_trials']['tmin_tmax'] = [-0.25, 2.75]

# ALL WORDS
    comparisons['all_words'] = {}
    comparisons['all_words']['queries'] = ["word_string.str.len()>1"]
    comparisons['all_words']['condition_names'] = ['All words']
    comparisons['all_words']['colors'] = ['b']
    comparisons['all_words']['sort'] = ['num_letters', 'word_position', 'word_string']
    comparisons['all_words']['y-tick-step'] = 40
  
# Sanity checks:
    comparisons['first_last_word'] = {}
    comparisons['first_last_word']['queries'] = ["word_position==1", "last_word==True"]
    comparisons['first_last_word']['condition_names'] = ['First_word', 'Last_word']
    comparisons['first_last_word']['colors'] = ['b', 'g']

# GRAMMATICAL NUMBER:
    comparisons['test'] = {}
    comparisons['test']['queries'] = ['sentence_length == 2 and last_word==1', 'sentence_length == 3 and last_word==1', 'sentence_length == 4 and last_word==1', 'sentence_length ==5 and last_word==1']
    comparisons['test']['condition_names'] = ['2', '3', '4', '5']
    comparisons['test']['colors'] = ['k', 'r', 'g', 'b']
    comparisons['test']['ls'] = ['-', '-', '-', '-']
    comparisons['test']['sort'] = ['word_string']
    # Nouns and verbs
    comparisons['number_nouns_verbs'] = {}
    comparisons['number_nouns_verbs']['queries'] = ["sentence_length<4 and Declarative==1 and word_position==2 and (word_string in ['boy', 'girl'])", 
                                                             "sentence_length<4 and Declarative==1 and word_position==2 and (word_string in ['boys', 'girls'])", 
                                                             "sentence_length<4 and Declarative==1 and word_position==3 and pos in ['VBZ']",
                                                             "sentence_length<4 and Declarative==1 and word_position==3 and pos in ['VB']"]
    comparisons['number_nouns_verbs']['condition_names'] = ['Noun Singular', 'Noun Plural', 'Verb Singular', 'Verb Plural', 'Pronoun Singular', 'Pronoun Plural', 'Verb-pro Singular', 'Verb-pro Plural']
    comparisons['number_nouns_verbs']['colors'] = ['r', 'b', 'r', 'b']
    comparisons['number_nouns_verbs']['ls'] = ['-', '-', '--', '--']
    comparisons['number_nouns_verbs']['sort'] = ['num_letters', 'word_string']
    
    # Pronouns and verbs
    comparisons['number_pronouns_verbs'] = {}
    comparisons['number_pronouns_verbs']['queries'] = ["sentence_length<3 and Declarative==1 and word_position==1 and (word_string in ['he', 'she'])", 
                                                       "sentence_length<3 and Declarative==1 and word_position==1 and (word_string in ['they'])", 
                                                       "sentence_length<3 and Declarative==1 and word_position==2 and pos in ['VBZ']", 
                                                       "sentence_length<3 and Declarative==1 and word_position==2 and pos in ['VBP']"]
    comparisons['number_pronouns_verbs']['condition_names'] = ['Noun Singular', 'Noun Plural', 'Verb Singular', 'Verb Plural', 'Pronoun Singular', 'Pronoun Plural', 'Verb-pro Singular', 'Verb-pro Plural']
    comparisons['number_pronouns_verbs']['colors'] = ['r', 'b', 'r', 'b']
    comparisons['number_pronouns_verbs']['ls'] = ['-', '-', '--', '--']
    comparisons['number_pronouns_verbs']['sort'] = ['num_letters', 'word_string']
    
    # Nouns:
    comparisons['grammatical_number_nouns'] = {}
    comparisons['grammatical_number_nouns']['queries'] = ["sentence_length<4 and word_position==2 and (word_string in ['boy', 'girl', 'man', 'woman', 'host', 'grandpa'])", "sentence_length<4 and word_position==2 and (word_string in ['boys', 'girls', 'men', 'women', 'actors'])"]
    comparisons['grammatical_number_nouns']['condition_names'] = ['Singular', 'Plural']
    comparisons['grammatical_number_nouns']['colors'] = ['b', 'g']

    # Verbs:
    #comparisons['grammatical_number_verbs'] = {}
    #comparisons['grammatical_number_verbs']['queries'] = []
    #comparisons['grammatical_number_verbs']['condition_names'] = ['Singular', 'Plural']
    #comparisons['grammatical_number_verbs']['colors'] = ['b', 'g']

    # Pronouns:
    comparisons['grammatical_number_pronouns'] = {}
    comparisons['grammatical_number_pronouns']['queries'] = ["sentence_length<3 and word_position==1 and (word_string in ['he', 'she'])", "sentence_length<3 and word_position==1 and (word_string in ['they'])"]
    comparisons['grammatical_number_pronouns']['condition_names'] = ['Singular', 'Plural']
    comparisons['grammatical_number_pronouns']['colors'] = ['b', 'g']

    # Pronouns (end of sentence):
    comparisons['grammatical_number_pronouns_end'] = {}
    comparisons['grammatical_number_pronouns_end']['queries'] = ["sentence_length<3 and (sentence_string.str.startswith('he') or sentence_string.str.startswith('she'))", "sentence_length<3 and (sentence_string.str.startswith('they'))"]
    comparisons['grammatical_number_pronouns_end']['condition_names'] = ['Singular', 'Plural']
    comparisons['grammatical_number_pronouns_end']['colors'] = ['b', 'g']
  
# Nouns vs. verbs
    comparisons['nouns_verbs'] = {}
    comparisons['nouns_verbs']['queries'] = ["word_position==2 and (pos=='NN' or pos=='NNS')", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG')"]
    comparisons['nouns_verbs']['condition_names'] = ['Noun', 'Verb']
    comparisons['nouns_verbs']['colors'] = ['b', 'g']
    comparisons['nouns_verbs']['sort'] = ['word_string']

# Gender
    
    comparisons['gender_pronouns'] = {}
    comparisons['gender_pronouns']['queries'] = ["word_position==1 and word_string=='he'", "word_position==1 and word_string=='she'"]
    comparisons['gender_pronouns']['condition_names'] = ['Masculine', 'Feminine']
    comparisons['gender_pronouns']['colors'] = ['b', 'g']

    comparisons['gender_nouns'] = {}
    comparisons['gender_nouns']['queries'] = ["word_position==2 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men')", "word_position==2 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women')"]
    comparisons['gender_nouns']['condition_names'] = ['Masculine', 'Feminine']
    comparisons['gender_nouns']['colors'] = ['b', 'g']


# Sentence type    
    comparisons['declarative_questions'] = {}
    comparisons['declarative_questions']['queries'] = ["Declarative==1 and word_position==-1", "Question==1 and word_position==-1"]
    comparisons['declarative_questions']['condition_names'] = ['Declarative', 'Question']
    comparisons['declarative_questions']['colors'] = ['b', 'g']
    
    comparisons['decl_quest_len2'] = {}
    comparisons['decl_quest_len2']['queries'] = ["Declarative==1 and word_position==1 and sentence_length==2", "Question==1 and word_position==1 and sentence_length==2"]
    comparisons['decl_quest_len2']['condition_names'] = ['Declarative', 'Question']
    comparisons['decl_quest_len2']['colors'] = ['b', 'g']
    comparisons['decl_quest_len2']['sort'] = ['word_string']

    
# Embedding
    comparisons['embedding'] = {}
    comparisons['embedding']['queries'] = ["Embedding==1 and word_position==-1", "Declarative==1 and sentence_length==5 and Embedding==0 and word_position==-1"]
    comparisons['embedding']['condition_names'] = ['Embedding', 'Long_declarative']
    comparisons['embedding']['colors'] = ['b', 'g']

# Number of letters
    comparisons['word_length'] = {}
    comparisons['word_length']['queries'] = "num_letters"
    comparisons['word_length']['condition_names'] = []
    comparisons['word_length']['colors'] = []

# Word string
    comparisons['word_string'] = {}
    comparisons['word_string']['queries'] = "word_string"
    comparisons['word_string']['condition_names'] = []
    comparisons['word_string']['colors'] = []
    comparisons['word_string']['tmin_tmax'] = [-1.2, 1.7]
    comparisons['word_string']['y-tick-step'] = 20
    
# Word string two -word sentences:
    comparisons['word_string_len2'] = {}
    comparisons['word_string_len2']['queries'] = ["sentence_length<3 and word_position==1 and word_string == 'he'", "sentence_length<3 and word_position==1 and word_string == 'she'", "sentence_length<3 and word_position==1 and word_string == 'they'", "sentence_length<3 and word_position==1 and word_string == 'who'" ]
    comparisons['word_string_len2']['condition_names'] = ['He', 'She', 'They', 'Who']
    comparisons['word_string_len2']['colors'] = ['r', 'g', 'b', 'm']
    comparisons['word_string_len2']['tmin_tmax'] = [-1.2, 1.7]
    comparisons['word_string_len2']['sort'] = ['sentence_string']
    comparisons['word_string_len2']['y-tick-step'] = 3

# Word Starts with
    letters = ['e', 'b', 'h', 'k', 's', 'v', 'j', 'f', 'p', 'm', 'd', 'a', 't', 'i', 'r', 'o', 'u', 'g', 'c', 'l', 'w']
    comparisons[f'word_startswith'] = {}
    comparisons[f'word_startswith']['queries'] = [f"word_string.str.startswith('{letter}')" for letter in letters]
    comparisons[f'word_startswith']['condition_names'] = letters
    comparisons[f'word_startswith']['colors'] = ['b'] * len(letters)

# Word Ends with
    letters = ['e', 'b', 'h', 'k', 's', 'v', 'j', 'f', 'p', 'm', 'd', 'a', 't', 'i', 'r', 'o', 'u', 'g', 'c', 'l', 'w']
    comparisons[f'word_endswith'] = {}
    comparisons[f'word_endswith']['queries'] = [f"word_string.str.endswith('{letter}')" for letter in letters]
    comparisons[f'word_endswith']['condition_names'] = letters
    comparisons[f'word_endswith']['colors'] = ['b'] * len(letters)

# Contains a letter
    letters = ['e', 'b', 'h', 'k', 's', 'v', 'j', 'f', 'p', 'm', 'd', 'a', 't', 'i', 'r', 'o', 'u', 'g', 'c', 'l', 'w']
    comparisons[f'word_contains'] = {}
    comparisons[f'word_contains']['queries'] = [f"word_string.str.contains('{letter}')" for letter in letters]
    comparisons[f'word_contains']['condition_names'] = letters
    comparisons[f'word_contains']['colors'] = ['b'] * len(letters)

# EOS effect
    comparisons['eos'] = {}
    comparisons['eos']['queries'] = ["last_word==True and (block in [1, 3, 5])", "last_word==True and (block in [2, 4, 6])"]
    comparisons['eos']['condition_names'] = ['Visual', 'Auditory'] 
    comparisons['eos']['colors'] = ['b', 'r'] 
    comparisons['eos']['sort'] = ['sentence_length', 'num_letters']
    comparisons['eos']['tmin_tmax'] = [-0.3, 1]
    comparisons['eos']['y-tick-step'] = 15

# WORD FREQ
    comparisons['word_zipf'] = {}
    comparisons['word_zipf']['queries'] = ["(word_zipf>=%i) and (word_zipf<%i) and (block in [1, 3, 5])"%(i, j) for (i, j) in zip(range(3, 7), range(4, 7))] 
    comparisons['word_zipf']['condition_names'] = [str(i-1)+'<zipf<'+str(i) for i in range(4, 7)]
    comparisons['word_zipf']['colors'] = []
    comparisons['word_zipf']['sort'] = ['num_letters']

# PART OF SPEECH (POS)
    comparisons['pos'] = {}
    comparisons['pos']['queries'] = "pos"
    comparisons['pos']['condition_names'] = []
    comparisons['pos']['colors'] = []

# PART OF SPEECH (POS) of DT/WP/PRP
    comparisons['pos_first'] = {}
    comparisons['pos_first']['queries'] = ["pos=='DT'", "pos=='WP'", "pos=='PRP'"]
    comparisons['pos_first']['condition_names'] = ["Determiner", "WH-question", "Pronoun"]
    comparisons['pos_first']['colors'] = ["b", "r", "g"]
    comparisons['pos_first']['sort'] = ['word_string']

# VERB TYPE
    comparisons['verb_type'] = {}
    comparisons['verb_type']['queries'] = ["word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Transitive==1", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Unergative==1", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Unaccusative==1", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Reflexive==1", "word_position==2 and (pos=='NN' or pos=='NNS')"]
    comparisons['verb_type']['condition_names'] = ['Transitive', 'Unergative', 'Unaccusative', 'Reflexive', 'Noun']
    comparisons['verb_type']['colors'] = ['b', 'c', 'r', 'm', 'k']
    comparisons['verb_type']['sort'] = ['word_string']
    
   
# Phonemes
    vowels, consonants = get_phone_classes()
    # PLACE
    comparisons['place_of_articulation'] = {}
    comparisons['place_of_articulation']['queries'] = []
    comparisons['place_of_articulation']['condition_names'] = []
    for cl in ['BILAB', 'LABDENT', 'DENT', 'ALV', 'PAL', 'LAR']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['"' + s.strip(' ') + '"' for s in curr_class]
        curr_query = "phone_string in [%s]" % (', '.join(curr_class))
        comparisons['place_of_articulation']['queries'].append(curr_query)
        comparisons['place_of_articulation']['condition_names'].append(cl)
    comparisons['place_of_articulation']['colors'] = ['r', 'g', 'b', 'y', 'c', 'm']
    comparisons['place_of_articulation']['sort'] = ['phone_string', 'word_position', 'phone_position']
    comparisons['place_of_articulation']['tmin_tmax'] = [-0.1, 0.4]
    comparisons['place_of_articulation']['y-tick-step'] = 60

    # MANNER
    comparisons['manner_of_articulation'] = {}
    comparisons['manner_of_articulation']['queries'] = []
    comparisons['manner_of_articulation']['condition_names'] = []
    for cl in ['STOP', 'FRIC', 'NAS', 'LIQ', 'SIB', 'GLI']:
        curr_class = consonants[cl].split(' ')
        #curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_class = ['"'+s.strip(' ')+'"' for s in curr_class]
        curr_query = "phone_string in [%s]" % (', '.join(curr_class))
        comparisons['manner_of_articulation']['queries'].append(curr_query)
        comparisons['manner_of_articulation']['condition_names'].append(cl)
    comparisons['manner_of_articulation']['colors'] = ['r', 'g', 'b', 'y', 'c', 'm']
    comparisons['manner_of_articulation']['sort'] = ['phone_string', 'word_position', 'phone_position']
    comparisons['manner_of_articulation']['tmin_tmax'] = [-0.1, 0.4]
    comparisons['manner_of_articulation']['y-tick-step'] = 60
    

    # SONORITY
    comparisons['sonority'] = {}
    comparisons['sonority']['queries'] = []
    comparisons['sonority']['condition_names'] = []
    for cl in ['VLS', 'VOI', 'SON']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['"' + s.strip(' ') + '"' for s in curr_class]
        curr_query = "phone_string in [%s]" % (', '.join(curr_class))
        comparisons['sonority']['queries'].append(curr_query)
        comparisons['sonority']['condition_names'].append(cl)
    comparisons['sonority']['colors'] = ['r', 'g', 'b']
    
    # CONSONANTS VS VOWELS
    comparisons['consonants_vowels'] = {}
    comparisons['consonants_vowels']['queries'] = []
    comparisons['consonants_vowels']['condition_names'] = ['Consonants', 'Vowels']
    all_consonants = ['"'+ph.strip()+'"' for cl in consonants.keys() for ph in consonants[cl].split(' ')]    
    curr_query = "phone_string in [%s]" % (', '.join(all_consonants))
    comparisons['consonants_vowels']['queries'].append(curr_query)
    all_vowels = ['"'+ph.strip()+'"' for cl in vowels.keys() for ph in vowels[cl].split(' ')]    
    curr_query = "phone_string in [%s]" % (', '.join(all_vowels))
    comparisons['consonants_vowels']['queries'].append(curr_query)
    comparisons['consonants_vowels']['colors'] = ['r', 'g']
    
    # PHONE STRING
    comparisons['phone'] = {}
    comparisons['phone']['queries'] = "phone_string"
    comparisons['phone']['condition_names'] = []
    comparisons['phone']['colors'] = [] 
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
    #CONSONANTS['COR'] = "TH DH T D S Z N CH JH SH ZH Y L R"
    #CONSONANTS['LAB'] = "P B M F V W"
    #CONSONANTS['DOR'] = "K G NG W"
    #CONSONANTS['BILAB'] = "P B M"
    #CONSONANTS['LABDENT'] = "F V"
    #CONSONANTS['DENT'] = "TH DH"
    #CONSONANTS['ALV'] = "T D S Z N"
    #CONSONANTS['PLV'] = "CH JH SH ZH"
    #CONSONANTS['PAL'] = "Y"
    #CONSONANTS['VEL'] = "K G NG"
    #CONSONANTS['LAR'] = "HH"
    # ADDED BY ME
    CONSONANTS['BILAB'] = "P B M"
    CONSONANTS['LABDENT'] = "F V"
    CONSONANTS['DENT'] = "TH DH"
    CONSONANTS['ALV'] = "T D S Z N CH JH SH ZH"
    CONSONANTS['PAL'] = "Y K G NG"
    CONSONANTS['LAR'] = "HH"

    ###CONSONANT MANNER
    CONSONANTS['STOP'] = "P T K B D G"
    #CONSONANTS['AFFR'] = "CH JH"
    CONSONANTS['FRIC'] = "F TH HH V DH"
    CONSONANTS['NAS'] = "M N NG"
    CONSONANTS['LIQ'] = "L R"
    CONSONANTS['SIB'] = "CH S SH JH Z ZH"
    CONSONANTS['GLI'] = "W Y"

    ###CONSONANT VOICING/SONORANCE
    CONSONANTS['VLS'] = "P T K CH F TH S SH HH"
    CONSONANTS['VOI'] = "B D G JH V DH Z ZH"
    CONSONANTS['SON'] = "M N NG L W R Y"

    return VOWELS, CONSONANTS
