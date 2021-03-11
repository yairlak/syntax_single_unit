import numpy as np


def get_features(metadata, feature_list):
    '''
    

    Parameters
    ----------
    metadata : TYPE
        DESCRIPTION.
    feature_list : TYPE
        DESCRIPTION.

    Returns
    -------
    design_matrices : TYPE
        DESCRIPTION.
    feature_values : TYPE
        DESCRIPTION.
    feature_info : TYPE
        DESCRIPTION.
    feature_groups : TYPE
        DESCRIPTION.

    '''
    if not feature_list:
        feature_list = ['letters', 'word_length', 'phone_string', 'is_first_word', 'is_last_word', 'word_position', 'tense', 'pos', 'pos_simple', 'word_zipf', 'morpheme', 'morph_complex', 'grammatical_number', 'embedding', 'wh_subj_obj', 'dec_quest', 'semantic_features', 'phonological_features']
    print(feature_list)
    num_samples = len(metadata.index)
    feature_values = []
    feature_info = {}
    # GROUP FEATURES
    feature_groups = {}
    feature_groups['orthography'] = ['letters', 'letter_by_position', 'word_length']
    feature_groups['phonology'] = ['phone_string', 'phonological_features']
    feature_groups['position'] = ['is_first_word', 'is_last_word', 'word_position']
    feature_groups['lexicon'] = ['tense', 'pos', 'pos_simple', 'word_zipf', 'morpheme', 'morph_complex']
    feature_groups['syntax'] = ['grammatical_number', 'gender', 'embedding', 'wh_subj_obj', 'dec_quest']
    feature_groups['semantics'] = ['semantic_features']

    # add '.' or '?' to end of word if needed (omitted in functions/read_logs..
    word_strings = np.asarray(metadata['word_string'])
    is_last_word = metadata['is_last_word']
    is_question = metadata['dec_quest']
    for i, (w, is_lw, is_q) in enumerate(zip(word_strings, is_last_word, is_question)):
        if is_lw:
            if is_q:
                word_strings[i] = w + '?'
            else:
                word_strings[i] = w + '.'
    ###########################            
    # BUILD THE DESIGN MATRIX #
    ###########################
    design_matrices = []
    for feature_name in feature_list:
        print(feature_name)
        dict_prop = get_feature_style(feature_name)
        #####################
        # SEMANTIC FEATURES #
        #####################
        if feature_name == 'semantic_features':
            values = metadata[feature_name]
            st = len(feature_values)
            values_unique = [feature_name + '-' + str(i) for i in range(1, 26)]
            feature_values.extend(values_unique)
            num_features = len(values_unique)
        
        #########################
        # PHONOLOGICAL FEATURES #
        #########################
        elif feature_name == 'phonological_features':
            values = metadata[feature_name]
            st = len(feature_values)
            values_unique = 'DORSAL,CORONAL,LABIAL,HIGH,FRONT,LOW,BACK,PLOSIVE,FRICATIVE,SYLLABIC,NASAL,VOICED,OBSTRUENT,SONORANT,SIBILANTS'.split(',')
            values_unique = [w.lower().capitalize() for w in values_unique]
            values_unique = [feature_name + '-' + w for w in values_unique]
            feature_values.extend(values_unique)
            num_features = len(values_unique)
            
        #####################
        # LETTER   FEATURES #
        #####################
        elif feature_name == 'letters':
            all_letters = []
            [all_letters.extend(set(w)) for w in metadata['word_string']]
            values_unique = sorted(list(set(all_letters)-set(['.', '?'])))
            print(values_unique)
            num_features = len(values_unique)
            st = len(feature_values) 
            
            values = []
            for w in metadata['word_string']:
                row_vector = np.zeros((1, num_features))
                curr_letters = list(set(w)-set(['.', '?']))
                IXs = [values_unique.index(l) for l in curr_letters]
                row_vector[0, IXs] = 1
                values.append(row_vector)
            values_unique = ['letter-' +  w for w in values_unique]
            feature_values.extend(values_unique)
        ######################
        # ALL OTHER FEATURES #
        ######################
        else:
            values = metadata[feature_name]
            values_unique = list(set(values))
            st = len(feature_values)
            if dict_prop['one-hot']: # ONE-HOT ENCODING OF FEATURE
                num_features = len(values_unique)
                values_unique = [feature_name +'-' + str(w) for w in values_unique]
                feature_values.extend(values_unique)
            else: # A SINGLE CONTINUOUS FEATURE
                num_features = 1
                feature_values.extend([feature_name])
                values_unique = [feature_name]
        
        ed = len(feature_values)
        design_matrix = np.zeros((num_samples, num_features))
        for i_sample, curr_value in enumerate(values):
            row_vector = np.zeros((1, num_features))
            if feature_name in ['semantic_features', 'letters', 'phonological_features']:
                row_vector = curr_value
            elif dict_prop['one-hot']:
                IX = values_unique.index(feature_name +'-' + str(curr_value))
                row_vector[0, IX] = 1
            else:
                row_vector[0,0] = curr_value
            design_matrix[i_sample, :] = row_vector
        #print(feature_name, dict_prop, row_vector)
        design_matrices.append(design_matrix)
        feature_info[feature_name] = {}
        feature_info[feature_name]['IXs'] = (st, ed)
        feature_info[feature_name]['color'] = dict_prop['color'] # For random coloring, leave empty (i.e., '')
        feature_info[feature_name]['ls'] = dict_prop['ls']
        feature_info[feature_name]['lw'] = dict_prop['lw']
        feature_info[feature_name]['names'] = values_unique
        

    # LUMP TOGETHER
    if len(design_matrices) > 1:
        design_matrices = np.hstack(design_matrices)
    else:
        design_matrices = design_matrices[0]
    return design_matrices, feature_values, feature_info, feature_groups


def get_feature_style(feature_name):
    dict_prop = {}

    if not dict_prop: # default style and setting
        dict_prop['color'] = 'grey'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
        
    #####################################
    # TENSE
    if feature_name == 'tense':
        dict_prop['color'] = 'xkcd:grass green'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    # WORD LOG-FREQUENCY
    if feature_name == 'word_zipf':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False

    # POS
    if feature_name == 'pos':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = '--'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    # POS SIMPLE
    if feature_name == 'pos_simple':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = '--'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    # MORPHEMES
    if feature_name == 'morpheme':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = '-.'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    # MORPHOLOGICAL COMPLEXITY
    if feature_name == 'morph_complex':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = '-.'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False

    #####################################
    # WORD POSITION
    if feature_name == 'word_position':
        dict_prop['color'] = 'grey'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    # LAST WORD
    if feature_name == 'is_last_word':
        dict_prop['color'] = 'grey'
        dict_prop['ls'] = '--'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    # FIRST WORD
    if feature_name == 'is_first_word':
        dict_prop['color'] = 'grey'
        dict_prop['ls'] = '-.'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    #####################################
    # GRAMMATICAL NUMBER
    if feature_name == 'grammatical_number':
        dict_prop['color'] = 'b'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    # EMBEDDING
    if feature_name == 'embedding':
        dict_prop['color'] = 'b'
        dict_prop['ls'] = '--'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    # SUBJECT VS OBJECT WH-QUESTIONS 
    if feature_name == 'wh_subj_obj':
        dict_prop['color'] = 'b'
        dict_prop['ls'] = '-.'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    # DECLARATIVE VS. QUESTIONS
    if feature_name == 'dec_quest':
        dict_prop['color'] = 'xkcd:bright blue'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    #####################################
    # WORD LENGTH
    if feature_name == 'word_length':
        dict_prop['color'] = 'r'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    # LETTERS
    if feature_name == 'letters':
        dict_prop['color'] = 'r'
        dict_prop['ls'] = '--'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    #####################################
    # PHONE STRING
    if feature_name == 'phone_string':
        dict_prop['color'] = 'm'
        dict_prop['ls'] = '--'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
    if feature_name == 'phonological_features':
        dict_prop['color'] = 'm'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
        
    #####################################
    # SEMANTICS
    if feature_name == 'semantic_features':
        dict_prop['color'] = 'xkcd:orange'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
	
    return dict_prop
