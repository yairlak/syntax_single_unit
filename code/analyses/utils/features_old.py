import os, glob, sys
import numpy as np
import mne
from scipy import io


def get_features(metadata, feature_list):
    '''
    feature_list (list): pos, letters, letter_by_position, bigrams, morphemes, first_letter, last_letter, word_length, sentence_length, word_position, word_zipf
    '''
    print(feature_list)
    num_samples = len(metadata.index)
    feature_names = []
    feature_groups = {}

    # add '.' or '?' to end of word if needed (omitted in functions/read_logs..
    word_strings = np.asarray(metadata['word_string'])
    is_last_word = metadata['last_word']
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
    for feature_name, dict_prop in feature_list.items():
        if feature_name in list(metadata):
            values = metadata[feature_name]
            values_unique = list(set(values))
            if dict_prop['one-hot']:
                num_features = len(values_unique)
            else:
                num_features = 1
            st = len(feature_names)
            feature_names.extend(feature_name)
            ed = len(feature_names) 
            feature_groups[feature_name] = {}
            feature_groups[feature_name]['IXs'] = (st, ed)
            feature_groups[feature_name]['color'] = dict_prop['color'] # For random coloring, leave empty (i.e., '')
            feature_groups[feature_name]['ls'] = dict_prop['ls']
            feature_groups[feature_name]['names'] = values_unique
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, curr_value in enumerate(values):
                row_vector = np.zeros((1, num_features))
                if dict_prop['one-hot']:
                    IX = values_unique.index(curr_value)
                    row_vector[0, IX] = 1
                else:
                    row_vector[0,0] = curr_value]
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)


        
        # TENSE
        if feature_set == 'tense':
            # LAST LETTER OF POS OF VERBS INDICATE THE TENSE (D - past, P - present, F - future, V - passive, I - infinitive-like, G - ing)
            all_pos = list(set(metadata['pos']))
            tenses = []
            dict_tense = {'D':'past', 'P':'present', 'F':'future', 'V':'passive', 'I':'inf_like', 'G':'ing'}
            for pos, word_string in zip(all_pos, word_strings):
                if pos.startswith('VB'):
                    tense = dict_tense[pos[-1]]
                    if tense == 'passive': tense = 'past' # HACK: all passive forms are in past
                else: # not a verb
                    tense = '-'
                tenses.append(tense)
            all_tenses = list(set(tenses))
            num_features = len(all_tenses)
            st = len(feature_names) 
            feature_names.extend(all_tenses)
            ed = len(feature_names) 
            feature_groups['tense'] = {}
            feature_groups['tense']['IXs'] = (st, ed)
            feature_groups['tense']['color'] = 'g' # For random coloring, leave empty (i.e., '')
            feature_groups['tense']['ls'] = '-'
            feature_groups['tense']['names'] = all_tenses
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, curr_tense in enumerate(tenses):
                row_vector = np.zeros((1, num_features))
                IX = all_tenses.index(curr_tense)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)

        
        # POS
        if feature_set == 'pos':
            all_pos = list(set(metadata['pos']))
            num_features = len(all_pos)
            st = len(feature_names) 
            feature_names.extend(all_pos)
            ed = len(feature_names) 
            feature_groups['pos'] = {}
            feature_groups['pos']['IXs'] = (st, ed)
            feature_groups['pos']['color'] = 'y' # For random coloring, leave empty (i.e., '')
            feature_groups['pos']['ls'] = '-'
            feature_groups['pos']['names'] = all_pos
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, curr_pos in enumerate(metadata['pos']):
                row_vector = np.zeros((1, num_features))
                IX = all_pos.index(curr_pos)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # POS SIMPLE
        if feature_set == 'pos_simple':
            pos = metadata['pos']
            pos = ['VB' if p.startswith('VB') else p for p in pos] # lump together all verbs (VBTRD, VBTRP, VBUEP,..)
            pos = ['NN' if p.startswith('NN') else p for p in pos] # lump together all nouns (NN, NNS)
            all_pos = list(set(pos))
            num_features = len(all_pos)
            st = len(feature_names) 
            feature_names.extend(all_pos)
            ed = len(feature_names) 
            feature_groups['pos_simple'] = {}
            feature_groups['pos_simple']['IXs'] = (st, ed)
            feature_groups['pos_simple']['color'] = 'y' # For random coloring, leave empty (i.e., '')
            feature_groups['pos_simple']['ls'] = '-'
            feature_groups['pos_simple']['names'] = all_pos
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, curr_pos in enumerate(pos):
                if curr_pos.startswith('VB'): curr_pos = 'VB'
                row_vector = np.zeros((1, num_features))
                IX = all_pos.index(curr_pos)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # LETTERS AS FEATURES
        if feature_set == 'letters':
            all_letters = []
            [all_letters.extend(set(w)) for w in word_strings]
            all_letters = sorted(list(set(all_letters)-set(['.', '?'])))
            print(all_letters)
            num_features = len(all_letters)
            st = len(feature_names) 
            feature_names.extend(all_letters)
            ed = len(feature_names) 
            feature_groups['letters'] = {}
            feature_groups['letters']['IXs'] = (st, ed)
            feature_groups['letters']['color'] = 'r'
            feature_groups['letters']['ls'] = '-'
            feature_groups['letters']['names'] = all_letters
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, w in enumerate(word_strings):
                row_vector = np.zeros((1, num_features))
                curr_letters = list(set(w)-set(['.', '?']))
                IXs = [all_letters.index(l) for l in curr_letters]
                row_vector[0, IXs] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # LETTERS AS FEATURES
        if feature_set == 'letter_by_position':
            # letters
            all_letters = []
            [all_letters.extend(set(w)) for w in word_strings]
            all_letters = sorted(list(set(all_letters)-set(['.', '?']))) # REMOVE ? and . (!!)
            num_letters = len(all_letters)
            print(all_letters)
            
            # build design matrix
            num_features = 3 * num_letters # multiply letter by 3, for first-middle-last
            st = len(feature_names) 
            feature_names.extend(all_letters)
            ed = len(feature_names) 
            feature_groups['letter_by_position'] = {}
            feature_groups['letter_by_position']['IXs'] = (st, ed)
            feature_groups['letter_by_position']['color'] = 'k'
            feature_groups['letter_by_position']['ls'] = '-'
            feature_groups['letter_by_position']['names'] = [l+'_'+str(p) for p in range(1, 4) for l in all_letters]
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, w in enumerate(word_strings):
                row_vector = np.zeros((1, num_features))
                curr_letters = list(set(w)-set(['.', '?']))

                w_without_marks = ''.join([i for i in w if i.isalpha()])
                for i_letter, letter in enumerate(w_without_marks):
                    if i_letter == 0: # is first letter in word
                        pos = 1
                    elif i_letter == len(w_without_marks)-1: # is last letter in word (excluding ? or .)
                        pos = 3
                    else:
                        pos = 2
                    IX = all_letters.index(letter) # find the number coding of the current letter identity
                    row_vector[0, IX + num_letters*(pos-1)] = 1
                    #print(IX, i_letter, pos, letter)
                #print(row_vector)
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # BIGRAMS
        if feature_set == 'bigrams':
            all_bigrams = []
            for w in word_strings:
                for i in range(len(w)-1):
                    curr_bigram = w[i]+w[i+1]
                    if not ('.' in curr_bigram or '?' in curr_bigram):
                        all_bigrams.append(curr_bigram)
            all_bigrams = sorted(list(set(all_bigrams)))
            print(all_bigrams)
            num_features = len(all_bigrams)
            st = len(feature_names) 
            feature_names.extend(all_bigrams)
            ed = len(feature_names) 
            feature_groups['bigrams'] = {}
            feature_groups['bigrams']['IXs'] = (st, ed)
            feature_groups['bigrams']['color'] = 'r'
            feature_groups['bigrams']['ls'] = '-.'
            feature_groups['bigrams']['names'] = all_bigrams
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, w in enumerate(word_strings):
                row_vector = np.zeros((1, num_features))
                curr_bigrams = [w[i]+w[i+1] for i in range(len(w)-1)]
                IXs = []
                for bi in curr_bigrams:
                    if not ('.' in bi or '?' in bi):
                        IXs.append(all_bigrams.index(bi))
                row_vector[0, IXs] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # MORPHEMES
        if feature_set == 'morphemes':
            all_morphemes = ['d', 'ed', 'y', 'es', 'ing', 's', '']
            st = len(feature_names) 
            feature_names.extend(all_morphemes)
            ed = len(feature_names) 
            feature_groups['morphemes'] = {}
            feature_groups['morphemes']['IXs'] = (st, ed)
            feature_groups['morphemes']['color'] = 'r'
            feature_groups['morphemes']['ls'] = '-.'
            feature_groups['morphemes']['names'] = ['morphemes']
            feature_names.extend(all_morphemes)
            num_features = len(all_morphemes)
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, m in enumerate(metadata['morpheme']):
                row_vector = np.zeros((1, num_features))
                if len(m)>0:
                    IX = all_morphemes.index(m)
                    row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # MORPHOLOGICAL COMPLEXITY
        if feature_set == 'morph_complex':
            st = len(feature_names) 
            feature_names.extend(['morph_complex'])
            ed = len(feature_names) 
            feature_groups['morph_complex'] = {}
            feature_groups['morph_complex']['IXs'] = (st, ed)
            feature_groups['morph_complex']['color'] = 'r'
            feature_groups['morph_complex']['ls'] = '-.'
            feature_groups['morph_complex']['names'] = ['morph_complex']
            morph_complex = [1 if m in ['d', 'ed', 'y', 'es', 'ing','s'] else 0 for m in metadata['morpheme']]
            design_matrix = np.asarray(morph_complex).reshape(-1, 1).astype(int)
            design_matrices.append(design_matrix)
        
        # FIRST LETTER
        if feature_set == 'first_letter':
            first_letters = [w[0] for w in word_strings]
            all_first_letters = sorted(list(set(first_letters)))
            st = len(feature_names) 
            feature_names.extend(all_first_letters)
            ed = len(feature_names) 
            feature_groups['first_letter'] = {}
            feature_groups['first_letter']['IXs'] = (st, ed)
            feature_groups['first_letter']['color'] = 'r'
            feature_groups['first_letter']['names'] = all_first_letters
            num_features = len(all_first_letters)
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, l in enumerate(first_letters):
                row_vector = np.zeros((1, num_features))
                IX = all_first_letters.index(l)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        # LAST LETTER
        if feature_set == 'last_letter':
            last_letters = []
            for w in word_strings:
                last_letter = w[-1]
                if last_letter in ['.', '?']:
                    last_letter = w[-2]
                last_letters.append(last_letter)
            all_last_letters = sorted(list(set(last_letters)))
            print(all_last_letters)
            st = len(feature_names) 
            feature_names.extend(all_last_letters)
            ed = len(feature_names) 
            feature_groups['last_letter'] = {}
            feature_groups['last_letter']['IXs'] = (st, ed)
            feature_groups['last_letter']['color'] = 'y'
            feature_groups['last_letter']['names'] = all_last_letters
            num_features = len(all_last_letters)
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, l in enumerate(last_letters):
                row_vector = np.zeros((1, num_features))
                IX = all_last_letters.index(l)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)

        # WORD LENGTH
        if feature_set == 'word_length':
            st = len(feature_names) 
            feature_names.extend(['word_length'])
            ed = len(feature_names) 
            feature_groups['word_length'] = {}
            feature_groups['word_length']['IXs'] = (st, ed)
            feature_groups['word_length']['color'] = 'g'
            feature_groups['word_length']['names'] = ['word_length']
            word_lengths = [len(w) for w in word_strings] # decompose each word into letters
            design_matrix = np.asarray(word_lengths).reshape(-1, 1)
            design_matrices.append(design_matrix)
        
        # SENTENCE LENGTH
        if feature_set == 'sentence_length':
            st = len(feature_names) 
            feature_names.extend(['sentence_length'])
            ed = len(feature_names) 
            feature_groups['sentence_length'] = {}
            feature_groups['sentence_length']['IXs'] = (st, ed)
            feature_groups['sentence_length']['color'] = 'g'
            feature_groups['sentence_length']['ls'] = '--'
            feature_groups['sentence_length']['names'] = ['sentence_length']
            sentence_lengths = metadata['sentence_length'] # decompose each word into letters
            design_matrix = np.asarray(sentence_lengths).reshape(-1, 1)
            design_matrices.append(design_matrix)
        
        # WORD POSITION
        if feature_set == 'word_position':
            st = len(feature_names) 
            feature_names.extend(['word_position'])
            ed = len(feature_names) 
            feature_groups['word_position'] = {}
            feature_groups['word_position']['IXs'] = (st, ed)
            feature_groups['word_position']['color'] = 'm'
            feature_groups['word_position']['ls'] = '-'
            feature_groups['word_position']['names'] = ['word_position']
            word_positions = metadata['word_position'] # decompose each word into letters
            design_matrix = np.asarray(word_positions).reshape(-1, 1)
            design_matrices.append(design_matrix)
        
        # GRAMMATICAL NUMBER
        if feature_set == 'grammatical_number':
            st = len(feature_names) 
            feature_names.extend(['grammatical_number'])
            ed = len(feature_names) 
            feature_groups['grammatical_number'] = {}
            feature_groups['grammatical_number']['IXs'] = (st, ed)
            feature_groups['grammatical_number']['color'] = 'm'
            feature_groups['grammatical_number']['ls'] = '-'
            feature_groups['grammatical_number']['names'] = ['grammatical_number']
            grammatical_numbers = metadata['grammatical_number'] # decompose each word into letters
            design_matrix = np.asarray(grammatical_numbers).reshape(-1, 1)
            design_matrices.append(design_matrix)
        
        # EMBEDDING
        if feature_set == 'embedding':
            st = len(feature_names) 
            feature_names.extend(['embedding'])
            ed = len(feature_names)
            # FIND STIMULUS NUMBER WITH THAT:
            stim_numbers_with_that = [] # LIST OF TUPLES (STIM_NUM, WORD_POSITION_OF_THAT)
            for w in word_string:
                if word_strings[IX] == 'that':
                    stim_numbers_with_that.append((metadata['stimulus_number'], metadata['word_position']))
            # GENERATE A LIST OF VALUES: 1 - IN EMBEDDING, 0 - IN MAIN
            embedding = []
            for curr_sn, curr_wp in zip(metadata['stimulus_number'], metadata['word_position']):
                is_in_embedding = any([1 if (curr_sn == sn and curr_wp>=wp) else 0 for (sn, wp) in stim_numbers_with_that])
                embedding.append(is_in_embedding)
                if embedding[-1]:
                    print(curr_sn, curr_wp, embedding[-1])
            #feature_groups['grammatical_number'] = {}
            #feature_groups['grammatical_number']['IXs'] = (st, ed)
            #feature_groups['grammatical_number']['color'] = 'm'
            #feature_groups['grammatical_number']['ls'] = '-'
            #feature_groups['grammatical_number']['names'] = ['grammatical_number']
            #grammatical_numbers = metadata['grammatical_number'] # decompose each word into letters
            #design_matrix = np.asarray(grammatical_numbers).reshape(-1, 1)
            #design_matrices.append(design_matrix)
        
        # WORD LOG-FREQUENCY
        if feature_set == 'word_zipf':
            st = len(feature_names) 
            feature_names.extend(['word_zipf'])
            ed = len(feature_names) 
            feature_groups['word_zipf'] = {}
            feature_groups['word_zipf']['IXs'] = (st, ed)
            feature_groups['word_zipf']['color'] = 'c'
            feature_groups['word_zipf']['names'] = ['word_zipf']
            design_matrix = np.asarray(metadata['word_zipf']).reshape(-1, 1)
            design_matrices.append(design_matrix)
        
        # LAST WORD
        if feature_set == 'is_last_word':
            st = len(feature_names) 
            feature_names.extend(['is_last_word'])
            ed = len(feature_names) 
            feature_groups['is_last_word'] = {}
            feature_groups['is_last_word']['IXs'] = (st, ed)
            feature_groups['is_last_word']['color'] = 'm'
            feature_groups['is_last_word']['ls'] = '--'
            feature_groups['is_last_word']['names'] = ['is_last_word']
            design_matrix = np.asarray(metadata['last_word']).reshape(-1, 1).astype(int)
            design_matrices.append(design_matrix)
        
        # FIRST WORD
        if feature_set == 'is_first_word':
            st = len(feature_names) 
            feature_names.extend(['is_first_word'])
            ed = len(feature_names) 
            feature_groups['is_first_word'] = {}
            feature_groups['is_first_word']['IXs'] = (st, ed)
            feature_groups['is_first_word']['color'] = 'm'
            feature_groups['is_first_word']['ls'] = '-.'
            feature_groups['is_first_word']['names'] = ['is_first_word']
            is_first_word = [1 if wp==1 else 0 for wp in metadata['word_position']]
            design_matrix = np.asarray(is_first_word).reshape(-1, 1)
            design_matrices.append(design_matrix)
        
            
        # WORD STRING
        if feature_set == 'word_string':
            all_word_strings = list(set(word_strings))
            st = len(feature_names) 
            feature_names.extend(['word_string'])
            ed = len(feature_names) 
            feature_groups['word_string'] = {}
            feature_groups['word_string']['IXs'] = (st, ed)
            feature_groups['word_string']['color'] = ''
            feature_groups['word_string']['names'] = ['word_string']
            feature_names.extend(all_word_strings)
            num_features = len(all_word_strings)
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, w in enumerate(word_strings):
                row_vector = np.zeros((1, num_features))
                IX = all_word_strings.index(w)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)
        
        
        
        # PREVIOUS WORD
        if feature_set == 'previous_word':
            previous_words = []
            for i_w, (w, pos, sent) in enumerate(zip(metadata['word_string'], metadata['word_position'], metadata['sentence_string'])):
                if pos>1:
                    previous_word = sent.split(' ')[pos-2]
                    previous_words.append(previous_word)
                elif pos==1:
                    previous_words.append('')
                else:
                    previous_words.append('-')
                print(previous_word, w, sent)

            all_previous_words = list(set(previous_words))
            st = len(feature_names) 
            feature_names.extend(all_previous_words)
            ed = len(feature_names) 
            feature_groups['previous_word'] = {}
            feature_groups['previous_word']['IXs'] = (st, ed)
            feature_groups['previous_word']['color'] = ''
            feature_groups['previous_word']['names'] = ['previous_word']
            num_features = len(all_previous_words)
            design_matrix = np.zeros((num_samples, num_features))
            for i_sample, w in enumerate(previous_words):
                row_vector = np.zeros((1, num_features))
                IX = all_previous_words.index(w)
                row_vector[0, IX] = 1
                design_matrix[i_sample, :] = row_vector
            design_matrices.append(design_matrix)

    # LUMP TOGETHER
    if len(design_matrices) > 1:
        design_matrices = np.hstack(design_matrices)
    else:
        design_matrices = design_matrices[0]
    return design_matrices, feature_names, feature_groups
