import numpy as np
import mne
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

class Features():
    def __init__(self, metadata, feature_list):
        self.metadata = metadata
        self.feature_list = feature_list

        # PRE-DEFINED GROUPED FEATURES
        features_groupped = {}
        features_groupped['positional'] = ['word_position', 'is_last_word']
        features_groupped['position'] = ['is_first_word', 'word_position', 'is_last_word']
        features_groupped['orthography'] = ['letters', 'word_length']
        features_groupped['orthography_pos'] = ['letter_by_position', 'word_length']
        features_groupped['phonology'] = ['phonological_features']
        features_groupped['phonemes'] = ['phone_string']
        features_groupped['lexicon'] = ['pos_simple', 'word_zipf',
                                        'morph_complex', 'tense']
        features_groupped['syntax'] = ['grammatical_number', 'gender', 'embedding',
                                       'wh_subj_obj', 'dec_quest',
                                       'syntactic_role', 'diff_thematic_role'] 
        #features_groupped['semantics'] = ['glove'] # try with a larger dim
        features_groupped['semantics'] = ['semantic_categories'] # Taken from Paradigm/word_features.docx
        features_groupped['word_string'] = ['word_string']
        self.features_groupped = features_groupped

    def add_punctuation(self):

        # add '.' or '?' to end of word (omitted in functions/read_logs..)
        word_strings = np.asarray(self.metadata['word_string'])
        is_last_word = self. metadata['is_last_word']
        is_question = self.metadata['dec_quest']
        for i, (w, is_lw, is_q) in enumerate(zip(word_strings,
                                                 is_last_word, is_question)):
            if is_lw:
                if is_q:
                    word_strings[i] = w + '?'
                else:
                    word_strings[i] = w + '.'
        self.metadata['word_string'] = word_strings

    def add_feature_info(self):
        # GET FEATURE NAMES, VALUES AND FIGURE-RELATED PROPERTIES (LS, LW)
        self.names = []
        feature_info = {}
        for feature_name in self.feature_list:
            # check if feature name is a group or a single feature
            if feature_name in self.features_groupped:
                features_to_loop = self.features_groupped[feature_name]
            else:
                features_to_loop = [feature_name]

            # COLLECT TOGETHER VALUES AND NAMES FROM ALL FEATURES IN GROUP
            values, names = [], []
            for feature in features_to_loop:
                dict_prop = get_feature_style(feature)
                feature_values, curr_names = \
                    get_feature_values(feature,
                                       self.metadata,
                                       dict_prop['one-hot'])
                print(feature, feature_values.shape, len(curr_names))
                names.extend(curr_names)
                if feature_values.ndim > 1:
                    feature_values = np.squeeze(feature_values)
                else:
                    feature_values = np.expand_dims(feature_values, axis=1)

                # SCALE CURR FEATURE
                if 'scale' in dict_prop:
                    if dict_prop['scale'] == 'standard':  # StandardScaler
                        scaler = StandardScaler()
                        print(f'Standard scaling {feature}')
                    else:  # a tuple with min and max val for scaling
                        min_val, max_val = dict_prop['scale']
                        scaler = MinMaxScaler(feature_range = (min_val, max_val))
                        print(f'MinMax scaling {feature} between {min_val} and {max_val}')
                else:  # default MinMaxScaler between 0 and 1
                    scaler = MinMaxScaler()
                    print(f'MinMax scaling {feature} between 0 and 1')
                feature_values = scaler.fit_transform(feature_values)
                values.append(feature_values)
            
            # LUMP TOGETHER VALUES AND NAMES FROM ALL FEATURES IN GROUP
            #values = [np.squeeze(A) if A.ndim > 1
            #          else np.expand_dims(A, axis=1)
            #          for A in values]
            values = np.hstack(values)
            feature_info[feature_name] = {}
            feature_info[feature_name]['names'] = names
            feature_info[feature_name]['values'] = values
            feature_info[feature_name]['color'] = dict_prop['color']
            feature_info[feature_name]['ls'] = dict_prop['ls']
            feature_info[feature_name]['lw'] = dict_prop['lw']
            self.names.extend(names)
            
        self.feature_info = feature_info
    
    def add_design_matrix(self):
        ###########################
        # BUILD THE DESIGN MATRIX #
        ###########################
        n_events = len(self.metadata)
        design_matrix = np.empty((n_events, 0))
        for feature_name in self.feature_list:
            # Add feature values to design matrix
            st = design_matrix.shape[1]
            X = self.feature_info[feature_name]['values']
            design_matrix = np.hstack((design_matrix, X))
            ed = design_matrix.shape[1]
            self.feature_info[feature_name]['IXs'] = (st, ed)
        self.design_matrix = design_matrix


    #def scale_design_matrix(self):
        # STANDARIZE THE FEATURE MATRIX #

    #    n_events = len(self.metadata)
    #    design_matrix = np.empty((n_events, 0))
    #    for feature_name in self.feature_list:
    #        st, ed = self.feature_info[feature_name]['IXs']
    #        X = self.design_matrix[:, st:ed]
    #        if 'scale' in self.feature_info[feature_name].keys():
    #            min_val, max_val = self.feature_info[feature_name]['scale']
    #        else:
    #            min_val, max_val = 0, 1
    #        print(f'Scaling {feature_name} between {min_val} and {max_val}')
    #        scaler = MinMaxScaler(feature_range = (min_val, max_val))
    #        X = scaler.fit_transform(X)
    #        design_matrix = np.hstack((design_matrix, X))
    #    self.design_matrix = design_matrix

    def add_raw_features(self, n_time_samples, sfreq):

        times_sec = self.metadata['event_time'].values
        times_samples = (times_sec * sfreq).astype(int)

        n_features = self.design_matrix.shape[1]
        X = np.zeros((n_time_samples, n_features))
        X[times_samples, :] = self.design_matrix

        # MNE-ize feature data
        ch_types = ['misc'] * len(self.names)
        info = mne.create_info(ch_names=self.names,
                               ch_types=ch_types,
                               sfreq=sfreq)
        raw_features = mne.io.RawArray(X.T, info)
        self.raw = raw_features


def get_feature_values(feature, metadata, one_hot):

    #####################
    # SEMANTIC FEATURES #
    #####################
    if feature == 'glove':
        values = metadata[feature]
        values = np.asarray([vec for vec in values])
        names = ['glove-' + str(i) for i in range(1, 26)]
    
    elif feature == 'semantic_categories':
        values = metadata[feature]
        values = np.stack(values)
        names = ['abstract', 'action', 'body', 'emotion', 'event', 'flower', 'food', 'fun', 'mental', 'movement', 'music', 'negative', 'object', 'perception', 'person', 'question', 'relation', 'search', 'sleep', 'speech', 'vehicle', 'water'] 
    
    #########################
    # PHONOLOGICAL FEATURES #
    #########################
    elif feature == 'phonological_features':
        values = metadata[feature]
        values = np.asarray([np.squeeze(vec) for vec in values])
        names = 'DORSAL,CORONAL,LABIAL,HIGH,FRONT,LOW,BACK,PLOSIVE,FRICATIVE,SYLLABIC,NASAL,VOICED,OBSTRUENT,SONORANT,SIBILANTS'.split(',')
        names = [w.lower().capitalize() for w in names]
        names = ['phono-' + w for w in names]

    elif feature == 'phone_string':
        all_phones = list(set(metadata['phone_string']))
        names = sorted(list(set(all_phones)-set(['', 'END_OF_WAV'])))
        num_features = len(names)

        values = []
        for phone in metadata['phone_string']:
            row_vector = np.zeros(num_features)
            if phone in names:
                IXs = names.index(phone)
                row_vector[IXs] = 1
            values.append(row_vector)
        names = ['Phone-' + p for p in names]



    #####################
    # LETTER   FEATURES #
    #####################
    elif feature == 'letters':
        all_letters = []
        [all_letters.extend(set(w)) for w in metadata['word_string']]
        names = sorted(list(set(all_letters)-set(['.', '?'])))
        num_features = len(names)

        values = []
        for w in metadata['word_string']:
            row_vector = np.zeros(num_features)
            curr_letters = list(set(w)-set(['.', '?']))
            IXs = [names.index(let) for let in curr_letters]
            row_vector[IXs] = 1
            values.append(row_vector)
        names = ['letter-' + w for w in names]

    elif feature == 'letter_by_position':
        values = metadata[feature]
        values = np.asarray([vec for vec in values])
        alphabet=[letter for letter in 'abcdefghijklmnopqrstuvwxyz']
        positions = ['First', 'Middle', 'Last']
        names = [letter + '-' + pos for pos in positions for letter in alphabet]
    
    
    ######################
    # ALL OTHER FEATURES #
    ######################
    else:
        values = metadata[feature]
        names = list(set(values))
        if one_hot:  # ONE-HOT ENCODING OF FEATURE
            values = []
            print(feature, names)
            names = [str(n) for n in names]
            if '0' in names: 
                names.remove('0')
                print(f'removed zero from {feature}, {names}')
            if '' in names: 
                names.remove('')
                print(f'removed empty string from {feature}, {names}')
            n_features = len(names)
            for i_event, curr_value in \
                    enumerate(metadata[feature]):
                row_vector = np.zeros((1, n_features))
                if str(curr_value) not in ['0', '']:
                    IX = names.index(str(curr_value))
                    row_vector[0, IX] = 1
                values.append(row_vector)
            if feature == 'word_string':
                feature = 'ws'
            names = [feature + '-' + str(name) for name in names]

        else:  # A SINGLE CONTINUOUS FEATURE
            names = [feature]
            values = metadata[feature]

    return np.asarray(values), names


def get_feature_style(feature_name):
    dict_prop = {}

    if not dict_prop:  # default style and setting, if not overwritten below
        dict_prop['color'] = 'grey'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False

    #####################################

    ##########
    # GROUPS #
    ##########

    # POSITION
    if feature_name == 'position':
        dict_prop['color'] = 'grey'
        dict_prop['ls'] = 'dashed'
        dict_prop['lw'] = 3

    # PHONOLOGY
    if feature_name == 'phonology':
        dict_prop['color'] = 'm'
        dict_prop['ls'] = 'dashdot'
        dict_prop['lw'] = 3

    # ORTHOGRAPHY
    if feature_name == 'orthography':
        dict_prop['color'] = 'r'
        dict_prop['ls'] = 'dashdot'
        dict_prop['lw'] = 3

    # LEXICON
    if feature_name == 'lexicon':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = 'dotted'
        dict_prop['lw'] = 3

    # SEMANTICS
    if feature_name == 'semantics':
        dict_prop['color'] = 'xkcd:orange'
        dict_prop['ls'] = 'solid'
        dict_prop['lw'] = 3

    # SYNTAX
    if feature_name == 'syntax':
        dict_prop['color'] = 'b'
        dict_prop['ls'] = (0, (3, 5, 1, 5, 1, 5)) #dashdotdotted
        dict_prop['lw'] = 3

    ############
    # FEATURES #
    ############

    # TENSE
    if feature_name == 'tense':
        dict_prop['color'] = 'xkcd:grass green'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
        #dict_prop['scale'] = (-1, 1)

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
    # SERIAL FEATURES
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
    
    # FIRST PHONE
    if feature_name == 'is_first_phone':
        dict_prop['color'] = 'm'
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
        #dict_prop['scale'] = (-1, 1)


    # GENDER
    if feature_name == 'gender':
        dict_prop['color'] = 'b'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
        #dict_prop['scale'] = (-1, 1)
    
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
        #dict_prop['scale'] = (-1, 1)
    
    # DECLARATIVE VS. QUESTIONS
    if feature_name == 'dec_quest':
        dict_prop['color'] = 'xkcd:bright blue'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    # NUMBER OF OPEN NODES
    if feature_name == 'n_open_nodes':
        dict_prop['color'] = 'xkcd:bright blue'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
    
    # SYNTACTIC ROLE
    if feature_name == 'syntactic_role':
        dict_prop['color'] = 'xkcd:bright blue'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
        #dict_prop['scale'] = (-1, 1)
    
    # DIFFERENT THEMATIC ROLE
    if feature_name == 'diff_thematic_role':
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
    if feature_name == 'word_string':
        dict_prop['color'] = 'g'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = True
    
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
    if feature_name == 'glove':
        dict_prop['color'] = 'xkcd:orange'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
        dict_prop['scale'] = 'standard'
    
    if feature_name == 'semantic_categories':
        dict_prop['color'] = 'xkcd:orange'
        dict_prop['ls'] = '-'
        dict_prop['lw'] = 3
        dict_prop['one-hot'] = False
	
    return dict_prop
