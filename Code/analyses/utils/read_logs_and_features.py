import numpy as np
import pickle, os
import math
from wordfreq import word_frequency, zipf_frequency
import pandas as pd
#from nltk import ngrams

def read_log(block, settings):
    '''

    :param block: (int) block number
    :param settings: class instance of settings
    :return: events (dict) with keys for event_times, block, phone/word/stimulus info
    '''
    log_fn = settings.log_name_beginning + str(block) + '.log'
    with open(os.path.join(settings.path2log, log_fn)) as f:
        lines = [l.strip('\n').split(' ') for l in f]

    events = {}
    if block in [2, 4, 6]:
        lines = [l for l in lines if l[1]=='PHONE_ONSET']
        events['event_time'] = [l[0] for l in lines]
        events['block'] = len(events['event_time']) * [block]
        events['first_phone'] = [int(l[2]) for l in lines]
        events['phone_position'] = [int(l[3]) for l in lines]
        events['phone_string'] = [l[6] for l in lines]
        events['word_position'] = [int(l[4]) for l in lines]
        events['word_string'] = [l[7] for l in lines]
        events['stimulus_number'] = [int(l[5]) for l in lines]

    elif block in [1, 3, 5]:
        lines = [l for l in lines if l[1] == 'DISPLAY_TEXT' and l[2] != 'OFF']
        events['event_time'] = [l[0] for l in lines]
        events['block'] = len(events['event_time']) * [block]
        events['first_phone'] = len(events['event_time']) * [-1] # not relevant for visual blocks
        events['phone_position'] = len(events['event_time']) * [-1] # not relevant for visual blocks
        events['phone_string'] = len(events['event_time']) * [-1]  # not relevant for visual blocks
        events['word_position'] = [int(l[4]) for l in lines]
        events['word_string'] = [l[5] for l in lines]
        events['stimulus_number'] = [int(l[3]) for l in lines]

    return events


def prepare_metadata(log_all_blocks, settings, params):
    '''

    :param log_all_blocks: list len = #blocks
    :param features: numpy
    :param settings:
    :param params:
    :return: metadata: list
    '''
    import pandas as pd

    word2features, word2features_new = load_word_features(settings)
    #print(word2features_new)
    num_blocks = len(log_all_blocks)

    # Create a dict with the following keys:
    keys = ['chronological_order', 'event_time', 'block', 'first_phone', 'phone_position', 'phone_string', 'stimulus_number',
            'word_position', 'word_string', 'pos', 'dec_quest', 'grammatical_number', 'wh_subj_obj',
            'word_length', 'sentence_string', 'sentence_length', 'last_word', 'morpheme', 'morpheme_type', 'word_type', 'word_freq', 'word_zipf']
    metadata = dict([(k, []) for k in keys])

    cnt = 1
    events_all_blocks = []
    for block, curr_block_events in log_all_blocks.items():
        for i in range(len(curr_block_events['event_time'])):
            sn = int(curr_block_events['stimulus_number'][i])
            wp = int(curr_block_events['word_position'][i])
            #print(sn, wp)
            #print(word2features_new[sn])
            metadata['stimulus_number'].append(sn)
            metadata['word_position'].append(wp)
            metadata['chronological_order'].append(cnt); cnt += 1
            metadata['event_time'].append((int(curr_block_events['event_time'][i]) - settings.time0) / 1e6)
            metadata['block'].append(curr_block_events['block'][i])
            is_first_phone = curr_block_events['first_phone'][i]
            metadata['first_phone'].append(is_first_phone)
            phone_pos = int(curr_block_events['phone_position'][i])
            metadata['phone_position'].append(phone_pos)
            metadata['phone_string'].append(curr_block_events['phone_string'][i])
            word_string = curr_block_events['word_string'][i]
            if word_string[-1] == '?' or word_string[-1] == '.':
                word_string = word_string[0:-1]
            word_string = word_string.lower()
            metadata['word_string'].append(word_string)
            word_freq = word_frequency(word_string, 'en')
            word_zipf = zipf_frequency(word_string, 'en')
            #print(word_string, type(word_freq), type(word_zipf))
            metadata['word_freq'].append(word_freq)
            metadata['word_zipf'].append(word_zipf)
            if curr_block_events['phone_string'][i] != 'END_OF_WAV' and phone_pos==1: # ADD FEATURES FROM XLS FILE
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(word2features_new[sn][wp]['word_length'])
                metadata['dec_quest'].append(word2features_new[sn][wp]['dec_quest'])
                metadata['grammatical_number'].append(word2features_new[sn][wp]['grammatical_number'])
                metadata['pos'].append(word2features_new[sn][wp]['pos'])
                metadata['wh_subj_obj'].append(word2features_new[sn][wp]['wh_subj_obj'])
                metadata['morpheme'].append(word2features[word_string][0])
                metadata['morpheme_type'].append(int(word2features[word_string][1]))
                metadata['word_type'].append(word2features[word_string][2])
            elif curr_block_events['phone_string'][i] != 'END_OF_WAV' and phone_pos>1 and is_first_phone: 
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(word2features_new[sn][wp]['word_length'])
                metadata['dec_quest'].append(-999)
                metadata['grammatical_number'].append(word2features_new[sn][wp]['grammatical_number'])
                metadata['pos'].append(word2features_new[sn][wp]['pos'])
                metadata['wh_subj_obj'].append(-999)
                metadata['morpheme'].append(word2features[word_string][0])
                metadata['morpheme_type'].append(int(word2features[word_string][1]))
                metadata['word_type'].append(word2features[word_string][2])
            elif curr_block_events['phone_string'][i] != 'END_OF_WAV' and not is_first_phone: 
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(-1)
                metadata['dec_quest'].append(-999)
                metadata['grammatical_number'].append(-999)
                metadata['pos'].append('-')
                metadata['wh_subj_obj'].append(-999)
                metadata['morpheme'].append('-')
                metadata['morpheme_type'].append('-')
                metadata['word_type'].append('-')
            else: # END-OF-WAV
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(-1)
                metadata['dec_quest'].append(-999)
                metadata['grammatical_number'].append(-999)
                metadata['pos'].append('-')
                metadata['wh_subj_obj'].append(-999)
                metadata['morpheme'].append('-')
                metadata['morpheme_type'].append('-')
                metadata['word_type'].append('-')
            
            metadata['last_word'].append(metadata['sentence_length'][-1] == metadata['word_position'][-1])
            
            # SINCE ONLY IN THE AUDITORY LOGS THERE'S END-OF-WAVE (WORD_POSITION=-1),
            # WE ADD HERE ANOTHER ROW FOR END OF SENTENCE FOR VISUAL BLOCKS (OFFSET OF LAST WORD)
            if metadata['last_word'][-1] and metadata['block'][-1] in [1, 3, 5]:
                metadata['chronological_order'].append(cnt); cnt += 1
                t = metadata['event_time'][-1] + params.word_ON_duration*1e-3
                metadata['event_time'].append(t)
                metadata['block'].append(block)
                metadata['first_phone'].append(0)
                metadata['phone_position'].append(-1)
                metadata['phone_string'].append('-')
                metadata['stimulus_number'].append(int(curr_block_events['stimulus_number'][i]))
                metadata['word_position'].append(-1)
                metadata['word_string'].append('.')
                metadata['pos'].append('-')
                metadata['morpheme'].append('-')
                metadata['morpheme_type'].append('-')
                metadata['word_type'].append('-')
                metadata['word_freq'].append(-999)
                metadata['word_zipf'].append(-999)
                metadata['sentence_string'].append(word2features_new[sn][wp]['sentence_string'])
                metadata['sentence_length'].append(word2features_new[sn][wp]['sentence_length'])
                metadata['word_length'].append(-1)
                metadata['dec_quest'].append(-999)
                metadata['grammatical_number'].append(-999)
                metadata['wh_subj_obj'].append(-999)
                metadata['last_word'].append(False)

    return pd.DataFrame(metadata)


def extend_metadata(metadata):
    ''' Add columns to metadata
    '''
    metadata = metadata.rename(columns={'last_word': 'is_last_word'})
    for i_w, w in enumerate(metadata['word_string']): # FIX ORTHOGRAPHIC MISTAKES
         if w.lower() == 'excercised': metadata.loc[i_w, 'word_string'] = 'exercised'
         if w.lower() == 'heared': metadata.loc[i_w, 'word_string'] = 'heard'
         if w.lower() == 'streched': metadata.loc[i_w, 'word_string'] = 'stretched'

    # TENSE
    # LAST LETTER OF POS OF VERBS INDICATE THE TENSE (D - past, P - present, F - future, V - passive, I - infinitive-like, G - ing)
    poss = metadata['pos']
    tenses = []
    dict_tense = {'D':'past', 'P':'present', 'F':'future', 'V':'passive', 'I':'inf_like', 'G':'ing'}
    for pos in poss:
        if pos.startswith('VB'):
            tense = dict_tense[pos[-1]]
            if tense == 'passive': tense = 'past' # HACK: all passive forms are in past
        else: # not a verb
            tense = '-'
        tenses.append(tense)
    metadata['tense'] = tenses
    
    # POS SIMPLE
    pos = metadata['pos']
    pos = ['VB' if p.startswith('VB') else p for p in pos] # lump together all verbs (VBTRD, VBTRP, VBUEP,..)
    pos = ['NN' if p.startswith('NN') else p for p in pos] # lump together all nouns (NN, NNS)
    metadata['pos_simple'] = pos

    # MORPHOLOGICAL COMPLEXITY
    morph_complex = [1 if m in ['d', 'ed', 'y', 'es', 'ing','s'] else 0 for m in metadata['morpheme']]
    metadata['morph_complex'] = morph_complex
  
    # IS FIRST WORD (LAST_WORD ALREADY IN METADATA) 
    is_first_word = [1 if wp==1 else 0 for wp in metadata['word_position']]
    metadata['is_first_word'] = is_first_word


    # EMBEDDING
    stim_numbers_with_that = [] # LIST OF TUPLES (STIM_NUM, WORD_POSITION_OF_THAT)
    for IX_word, w in enumerate(metadata['word_string']): # FIND STIMULUS NUMBER WITH THAT:
        if w == 'that':
            stim_numbers_with_that.append((metadata['stimulus_number'].tolist()[IX_word], metadata['word_position'].tolist()[IX_word]))
    embedding = [] # GENERATE A LIST OF VALUES: 1 - IN EMBEDDING, 0 - IN MAIN
    for curr_sn, curr_wp in zip(metadata['stimulus_number'], metadata['word_position']):
        is_in_embedding = any([1 if (curr_sn == sn and (curr_wp>=wp or curr_wp==-1)) else 0 for (sn, wp) in stim_numbers_with_that])
        #print(curr_sn, curr_wp, is_in_embedding)
        embedding.append(is_in_embedding)
    metadata['embedding'] = embedding

    # SEMANTIC FEATURES
    fn_glove = '../../../Paradigm/small_glove.twitter.27B.25d.txt'

    glove = load_glove_model(fn_glove)
    #print(sorted(glove.keys()))
    X = []
    for i_w, w in enumerate(metadata['word_string']):
        if list(metadata['word_length'])[i_w]>1:
            vec = glove[w]
        else:
            vec = np.zeros(25)
        X.append(vec)
    metadata['semantic_features'] = X            
    
    # PHONOLOGICAL FEATURES
    phones = metadata['phone_string']
    fn_phonologica_features = '../features/phone.csv'
    df_phonological_features = pd.read_csv(fn_phonologica_features)
    phonological_features = list(df_phonological_features)
    phonological_features.remove('PHONE')
    # for phonological_feature in phonological_features:
    #     print(phonological_feature)
    feature_values = []
    for ph in phones:
        if ph not in [-1, 'END_OF_WAV']:
            ph = ''.join([s for s in ph if not s.isdigit()]) # remove digits at the end if exist
            # feature_value = df_phonological_features.loc[df_phonological_features['PHONE'] == ph][phonological_feature]
            feature_value = df_phonological_features.loc[df_phonological_features['PHONE'] == ph]
            feature_values.append(feature_value.values[0][1::])
        else:
            feature_values.append(np.zeros((1, len(phonological_features))))
    metadata['phonological_features'] = feature_values
    # feature_values = np.vstack(feature_values)
    # feature_values = pd.DataFrame(data=feature_values, columns=phonological_features)
    # metadata = pd.concat((metadata, feature_values), axis=1)
    
    return metadata


def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """

    with open(glove_file, 'r', encoding='utf-8') as f:
        vectors = f.readlines()
    model = {}
    for line in vectors:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    return model


def get_sentences_start_end_length(SENTENCE_NUM_ORDER, settings):
    # Load text containing all sentences
    with open(os.path.join(settings.path2stimuli, settings.stimuli_text_file), 'r') as f:
        stimuli_sentences = f.readlines()
    sentences_length = [len(s.split(' ')) for s in stimuli_sentences]
    IX = [i-1 for i in SENTENCE_NUM_ORDER] # shift to count from zero
    sentences_length = np.asarray(sentences_length)[IX] #reorder array according to the (random) order of current block
    sentences_end = np.cumsum(sentences_length)
    sentences_start = [e-l+1 for (e, l) in zip(sentences_end, sentences_length)]

    sentences_length = dict(zip(range(1, len(sentences_length) + 1, 1), sentences_length))
    sentences_end = dict(zip(range(1, len(sentences_end) + 1, 1), sentences_end))
    sentences_start = dict(zip(range(1, len(sentences_start) + 1, 1), sentences_start))

    return sentences_start, sentences_end, sentences_length


def load_features(settings):
    import pandas

    # Read features file ('xlsx')
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, settings.features_file))
    headers = sheet.columns
    fields = []
    for i, header in enumerate(headers):
        fields.append(sheet[header].values)
    features = {'headers': headers, 'fields': fields}

    return features


def extract_comparison(comparison_list, features, settings, preferences):
    trial_numbers = features['fields'][0][1::]
    stimuli = features['fields'][1][1::]
    features = features['fields'][2::]
    contrast_names = comparison_list['fields'][0]

    comparisons = []

    ### Comparisons
    for i, contrast_name in enumerate(contrast_names):
        if preferences.use_metadata_only:
            # blocks_list = comparison_list['fields'][5][settings.comparisons][i].split(';')
            # align_to_list = comparison_list['fields'][4][settings.comparisons][i].split(';')
            blocks = comparison_list['fields'][4][i]
            align_to = comparison_list['fields'][3][i]
            generalize_to_modality = comparison_list['fields'][7][i]
            generalize_to_contrast = comparison_list['fields'][8][i]
            # for b, blocks in enumerate(blocks_list):
            #     for align_to in align_to_list:
            curr_dict = {}
            curr_dict['contrast_name'] = contrast_name + '_' + str(blocks) + '_' + align_to
            curr_dict['contrast'] = comparison_list['fields'][1][i]
            curr_query = curr_dict['contrast'][1:-1].split(',')
            curr_query = [s.strip() for s in curr_query]
            curr_dict['query'] = curr_query
            cond_labels = comparison_list['fields'][2][i]
            curr_dict['cond_labels'] = cond_labels[1:-1].split(',')
            curr_dict['align_to'] = align_to
            curr_dict['blocks'] = blocks
            curr_dict['generalize_to_blocks'] = generalize_to_modality
            curr_dict['generalize_to_contrast'] = generalize_to_contrast
            sortings = comparison_list['fields'][5][i]
            if isinstance(sortings, str):
                curr_dict['sorting'] = sortings.split(',')
            else:
                curr_dict['sorting'] = []
            curr_dict['union_or_intersection'] = comparison_list['fields'][6][i]

            comparisons.append(curr_dict)

        else:
            print('Metadata is not used')

    return comparisons


def load_POS_tags(settings):
    with open(os.path.join(settings.path2stimuli, settings.word2pos_file), 'rb') as f:
        word2pos = pickle.load(f)
        word2pos['exercised'] = word2pos['excercised']
        word2pos['stretched'] = word2pos['streched']
    return word2pos

def load_word_features(settings, word_features_filename='word_features.xlsx', word_features_filename_new = 'word_features_new.xlsx'):
    import pandas
    word2features = {}
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, word_features_filename))
    words = sheet['word_string']
    morphemes = sheet['morpheme']
    morpheme_types = sheet['morpheme_type']
    word_type = sheet['word_type'] # function or content word

    for w, m, t, cf in zip(words, morphemes, morpheme_types, word_type):
        if np.isnan(t):
            t=0
        if not isinstance(m, str):
            m=''
        word2features[w.lower()] = (m, t, cf)


    word2features['exercised'] = word2features['excercised']
    word2features['stretched'] = word2features['streched']
    
    ##
    word2features_new = {}
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, word_features_filename_new))
    sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
    for i, row in sheet.iterrows():
        s = row['stimulus_number']
        w = row['word_position']
        #print(s, w)
        #print(row)
        if s not in word2features_new.keys():
            word2features_new[s]={}
        if w not in word2features_new[s].keys():
            word2features_new[s][w]= {}
        for f in row.keys():
            if f in ['sentence_string', 'word_string', 'pos']:
                word2features_new[s][w][f] = row[f]
            else:
                word2features_new[s][w][f] = int(row[f])


        # add for word_position=-1 (end of sentence):
        if -1 not in word2features_new[s].keys():
            word2features_new[s][-1]= {}
            word2features_new[s][-1]['sentence_string'] = row['sentence_string']
            for f in row.keys():
                if f in ['word_string', 'pos']:
                    word2features_new[s][-1][f] = ''
                elif f == 'sentence_string':
                    pass
                else:
                    word2features_new[s][-1][f] = -999

    return word2features, word2features_new


def load_comparisons_and_features(settings):
    import pandas

    # Read comparison file ('xlsx')
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, settings.comparisons_file))
    headers = sheet.columns
    fields = []
    for i, header in enumerate(headers):
        fields.append(sheet[header].values)
        comparison_list = {'headers':headers, 'fields':fields}

    del sheet, headers

    # Read features file ('xlsx')
    sheet = pandas.read_excel(os.path.join(settings.path2stimuli, settings.features_file))
    headers = sheet.columns
    fields = []
    for i, header in enumerate(headers):
        fields.append(sheet[header].values)
    features = {'headers': headers, 'fields': fields}
    
    return comparison_list, features


class LogSingleUnit:
    def __init__(self, settings, block):
        self.log_filename = settings.log_name_beginning + str(block) + '.log'

    def append_log(self):
        with open(os.path.join(settings.path2data, self.log_filename)) as f:
            self.log_content = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            self.log_content = [x.strip() for x in self.log_content]

    def read_and_parse_log(self, settings):
        with open(os.path.join(settings.path2log, self.log_filename)) as f:
            log_content = [line.split() for line in f]

        # Find all event types (DISPLAY_TEXT, FIXATION, KEY_PRESS, etc.)
        event_types_in_paradigm_log = [i[1] for i in log_content]
        event_types_in_paradigm_log = list(set().union(event_types_in_paradigm_log, event_types_in_paradigm_log))

        # For each event type, extract onset times and stimulus information
        event_types_added = []
        for event_type in event_types_in_paradigm_log:
            if event_type == 'DISPLAY_TEXT':
                setattr(self, event_type + '_WORDS_ON_TIMES',
                        [i[0] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append(event_type + '_WORDS_ON_TIMES')
                setattr(self, event_type + '_WORD_NUM',
                        [i[2] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append(event_type + '_WORD_NUM')
                setattr(self, event_type + '_SENTENCE_NUM',
                        [i[3] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append(event_type + '_SENTENCE_NUM')
                setattr(self, event_type + '_WORD_SERIAL_NUM',
                        [i[4] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append(event_type + '_WORD_SERIAL_NUM')
                setattr(self, event_type + '_WORD_STRING',
                        [i[5] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append(event_type + '_WORD_STRING')
                setattr(self, 'FIRST_WORD_TIMES',
                        [i[0] for i in log_content if event_type == i[1] and i[2] != 'OFF' if i[4] == '1'])
                event_types_added.append('FIRST_WORD_TIMES')
                setattr(self, 'SENTENCE_NUM', [i[3] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append('SENTENCE_NUM')
                setattr(self, 'WORD_SERIAL_NUM', [i[4] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append('WORD_SERIAL_NUM')
                self.SENTENCE_NUM_ORDER = [];
                last_sent = []
                for i in self.SENTENCE_NUM:
                    if i != last_sent:
                        self.SENTENCE_NUM_ORDER.append(int(i))
                        last_sent = i
                event_types_added.append('SENTENCE_NUM_ORDER')
                setattr(self, event_type + '_OFF_TIMES',
                        [i[0] for i in log_content if event_type == i[1] and i[2] == 'OFF'])  # WORD-IMAGE IS 'OFF'
                event_types_added.append(event_type + '_OFF_TIMES')

                setattr(self, 'WORD_STRING', [i[5] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append('WORD_STRING')
                setattr(self, 'WORDS_ON_TIMES', [i[0] for i in log_content if event_type == i[1] and i[2] != 'OFF'])
                event_types_added.append('WORDS_ON_TIMES')

                sentences_start, sentences_end, sentences_length = get_sentences_start_end_length(
                    self.SENTENCE_NUM_ORDER, settings)
                setattr(self, 'FIRST_WORD_TIMES1', [i[0] for i in log_content if event_type == i[1] and i[2] != 'OFF' if
                                                    int(i[2]) in sentences_start.values()])
                event_types_added.append('FIRST_WORD_TIMES1')  # SANITY CHECK: FIRST_WORD_TIMES=FIRST_WORD_TIMES1
                setattr(self, 'LAST_WORD_TIMES', [i[0] for i in log_content if event_type == i[1] and i[2] != 'OFF' if
                                                  int(i[2]) in sentences_end.values()])
                event_types_added.append('LAST_WORD_TIMES')

                word_strings_parsed = [s[0:-1] if s[-1] in ['.', '?'] else s for s in self.WORD_STRING]
                num_letters = [len(s) for s in word_strings_parsed]
                setattr(self, 'num_letters', num_letters)
                event_types_added.append('num_letters')

            elif event_type == 'AUDIO_PLAYBACK_ONSET':
                setattr(self, event_type + '_WORDS_ON_TIMES',
                        [i[0] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append(event_type + '_WORDS_ON_TIMES')
                setattr(self, event_type + '_WORD_NUM',
                        [i[2] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append(event_type + '_WORD_NUM')
                setattr(self, event_type + '_WAV_FILE',
                        [i[3] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append('SENTENCE_NUM')
                setattr(self, event_type + '_WORD_SERIAL_NUM',
                        [i[4] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append(event_type + '_WORD_SERIAL_NUM')
                setattr(self, event_type + '_WORD_STRING',
                        [i[5] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append(event_type + '_WORD_STRING')

                setattr(self, 'FIRST_WORD_TIMES',
                        [i[0] for i in log_content if event_type == i[1] and i[2] != '_' if i[4] == '1'])
                event_types_added.append('FIRST_WORD_TIMES')
                setattr(self, 'SENTENCE_NUM',
                        [i[3].split('.')[0] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append(event_type + '_SENTENCE_NUM')
                setattr(self, 'WORD_SERIAL_NUM', [i[4] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append('WORD_SERIAL_NUM')

                self.SENTENCE_NUM_ORDER = [];
                last_sent = []
                for i in self.SENTENCE_NUM:
                    if i != last_sent:
                        self.SENTENCE_NUM_ORDER.append(int(i))
                        last_sent = i
                event_types_added.append('SENTENCE_NUM_ORDER')

                setattr(self, 'END_WAV_TIMES', [i[0] for i in log_content if event_type == i[1] and i[2] == '_'])
                event_types_added.append('END_WAV_TIMES')

                setattr(self, 'WORD_STRING', [i[5] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append('WORD_STRING')

                sentences_start, sentences_end, sentences_length = get_sentences_start_end_length(
                    self.SENTENCE_NUM_ORDER, settings)
                setattr(self, 'FIRST_WORD_TIMES1', [i[0] for i in log_content if event_type == i[1] and i[2] != '_' if
                                                    int(i[2]) in sentences_start.values()])
                event_types_added.append('FIRST_WORD_TIMES1')  # SANITY CHECK: FIRST_WORD_TIMES=FIRST_WORD_TIMES1
                setattr(self, 'LAST_WORD_TIMES', [i[0] for i in log_content if event_type == i[1] and i[2] != '_' if
                                                  int(i[2]) in sentences_end.values()])
                event_types_added.append('LAST_WORD_TIMES')
                setattr(self, 'WORDS_ON_TIMES', [i[0] for i in log_content if event_type == i[1] and i[2] != '_'])
                event_types_added.append('WORDS_ON_TIMES')

                setattr(self, 'sentences_start', sentences_start)
                setattr(self, 'sentences_end', sentences_end)
                setattr(self, 'sentences_length', sentences_length)

                word_strings_parsed = [s[0:-1] if s[-1] in ['.', '?'] else s for s in
                                       self.AUDIO_PLAYBACK_ONSET_WORD_STRING]

                num_letters = [len(s) for s in word_strings_parsed]
                setattr(self, 'num_letters', num_letters)
                event_types_added.append('num_letters')

            elif event_type == 'KEY_PRESS':
                setattr(self, event_type + '_SPACE_TIMES',
                        [i[0] for i in log_content if event_type == i[1] and i[2] == 'space'])
                event_types_added.append(event_type + '_SPACE_TIMES')
                list_of_key_press_times = [];
                previous_key_press_time = 0.
                for cnt, i in enumerate(log_content):
                    if event_type == i[1] and i[2] == 'l' and np.abs(previous_key_press_time - float(
                            i[0])) > 1e6:  # only if time between two key presses is greater than 1sec
                        list_of_key_press_times.append(i[0])
                        previous_key_press_time = float(i[0])
                setattr(self, event_type + '_l_TIMES', list_of_key_press_times)
                event_types_added.append(event_type + '_l_TIMES')

                # sentences_start, sentences_end, sentences_length = get_sentences_start_end_length(self.SENTENCE_NUM_ORDER, settings)
                # setattr(self, 'sentences_start', sentences_start)
                # setattr(self, 'sentences_end', sentences_end)
                # setattr(self, 'sentences_length', sentences_length)

            else:
                setattr(self, event_type + '_TIMES', [i[0] for i in log_content if event_type == i[1]])
                event_types_added.append(event_type + '_TIMES')

        setattr(self, 'event_types', event_types_added)

        return self
