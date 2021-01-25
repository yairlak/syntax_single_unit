def comparison_list():
    comparisons = {}

# Phones
    vowels, consonants = get_phone_classes()
    
    # PLACE
    comparisons[1] = {}
    comparisons[1]['name'] = 'place_of_articulation'
    comparisons[1]['train_queries'] = []
    comparisons[1]['train_condition_names'] = []
    for cl in ['COR', 'LAB', 'DOR']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[1]['train_queries'].append(curr_query)
        comparisons[1]['train_condition_names'].append(cl)
    comparisons[1]['colors'] = ['r', 'g', 'b']

    # MANNER
    comparisons[2] = {}
    comparisons[2]['name'] = 'manner_of_articulation'
    comparisons[2]['train_queries'] = []
    comparisons[2]['train_condition_names'] = []
    for cl in ['STOP', 'AFFR', 'FRIC', 'NAS', 'LIQ', 'SIB', 'GLI']:
        curr_class = consonants[cl].split(' ')
        #curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_class = ['"'+s.strip(' ')+'"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[2]['train_queries'].append(curr_query)
        comparisons[2]['train_condition_names'].append(cl)
    comparisons[2]['colors'] = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    # SONORITY
    comparisons[3] = {}
    comparisons[3]['name'] = 'sonority'
    comparisons[3]['train_queries'] = []
    comparisons[3]['train_condition_names'] = []
    for cl in ['VLS', 'VOI', 'SON']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[3]['train_queries'].append(curr_query)
        comparisons[3]['train_condition_names'].append(cl)
    comparisons[3]['colors'] = ['r', 'g', 'b']
    
    # BACKNESS and HEIGHT 
    comparisons[4] = {}
    comparisons[4]['name'] = 'backness_height'
    comparisons[4]['train_queries'] = []
    comparisons[4]['train_condition_names'] = []
    for cl in ['LOW', 'MID', 'HI', 'BCK', 'CNT', 'FRO', 'DIPH']:
        curr_class = vowels[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[4]['train_queries'].append(curr_query)
        comparisons[4]['train_condition_names'].append(cl)
    comparisons[4]['colors'] = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    # ROUNDEDNESS
    comparisons[5] = {}
    comparisons[5]['name'] = 'roundedness'
    comparisons[5]['train_queries'] = []
    comparisons[5]['train_condition_names'] = []
    for cl in ['RND', 'UNR']:
        curr_class = vowels[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[5]['train_queries'].append(curr_query)
        comparisons[5]['train_condition_names'].append(cl)
    comparisons[5]['colors'] = ['r', 'g']

    # TENSENESS
    comparisons[6] = {}
    comparisons[6]['name'] = 'tenseness'
    comparisons[6]['train_queries'] = []
    comparisons[6]['train_condition_names'] = []
    for cl in ['TNS', 'LAX']:
        curr_class = vowels[cl].split(' ')
        curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_query = "phone_string in [%s] and block in [2, 4, 6]" % (', '.join(curr_class))
        comparisons[6]['train_queries'].append(curr_query)
        comparisons[6]['train_condition_names'].append(cl)
    comparisons[6]['colors'] = ['r', 'g']

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
