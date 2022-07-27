import os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='554_13', type=str)
parser.add_argument('--hospital', default='UCLA', type=str)
parser.add_argument('--blocks', action='append', default=[], type=str)
args = parser.parse_args()

if not args.blocks:
    #args.blocks = [2]
    args.blocks = [1, 2, 3, 4, 5, 6]

args.patient = 'patient_' + args.patient

# Paths
path2logs = os.path.join('..', '..', '..', 'Data', args.hospital, args.patient, 'Logs')
logs_fns = glob.glob(os.path.join(path2logs, 'events_log_in_cheetah_clock_part*.log'))
logs_fns = [fn for fn in logs_fns if int(os.path.basename(fn)[32:33]) in args.blocks]

path2stimuli = os.path.join('..', '..', '..', 'Paradigm', 'Stimuli', 'Audio', 'normalized', 'resampled_16k')


def get_curr_words_and_phones(curr_line, path2stimuli):
    curr_wave_fn = curr_line[2]
    curr_stimulus_num = curr_wave_fn[0:-4]  # str

    lab_fn = os.path.join(path2stimuli, curr_stimulus_num + '.lab')  # contains sentence string
    with open(lab_fn, 'r') as f_lab:
        sentence = f_lab.readlines()
        sentence = sentence[0].strip('\n')
        words = [w.lower() for w in sentence.split(' ')]

    textgrid_fn = os.path.join(path2stimuli, curr_stimulus_num + '.TextGrid')  # contains word and phoneme onsets
    # ---------------------
    # get phonetic parsing
    # ---------------------
    phones = []
    f_textgrid = open(textgrid_fn, 'r')
    l = f_textgrid.readline()
    while l:
        if 'name = "phones"' in l:
            for _ in range(4):  # roll three lines
                l = f_textgrid.readline()
            while not 'item' in l:
                assert 'intervals' in l
                curr_phone = {}
                l = f_textgrid.readline()
                curr_phone['xmin'] = float(l.strip('\n').split(' ')[-1])
                l = f_textgrid.readline()
                curr_phone['xmax'] = float(l.strip('\n').split(' ')[-1])
                l = f_textgrid.readline()
                curr_phone['text'] = l.strip('\n').split(' ')[-1][1:-1]
                phones.append(curr_phone)
                l = f_textgrid.readline()  # roll one more line and assert position
        elif 'name = "words"' in l:
            break
        l = f_textgrid.readline()
    f_textgrid.close()

    # ---------------------
    # get word parsing
    # ---------------------
    words = []
    f_textgrid = open(textgrid_fn, 'r')
    l = f_textgrid.readline()
    while l:
        if 'name = "words"' in l:
            for _ in range(4):  # roll three lines
                l = f_textgrid.readline()
            while l:
                assert 'intervals' in l
                curr_phone = {}
                l = f_textgrid.readline()
                curr_phone['xmin'] = float(l.strip('\n').split(' ')[-1])
                l = f_textgrid.readline()
                curr_phone['xmax'] = float(l.strip('\n').lstrip('"').split(' ')[-1])
                l = f_textgrid.readline()
                curr_phone['text'] = l.strip('\n').split(' ')[-1][1:-1]
                words.append(curr_phone)
                l = f_textgrid.readline()  # roll one more line and assert position
        l = f_textgrid.readline()  # roll one more line
    f_textgrid.close()
    return words, phones


def add_words_to_which_phones_belong(words, phones):
    phones_with_words_to_which_they_belong = []
    last_in_word = ''
    for i_phone, phone_dict in enumerate(phones):
        new_phone_dict = {}
        new_phone_dict['in_word'] = []
        new_phone_dict['word_number'] = []
        for w, word_dict in enumerate(words):
            if phone_dict['xmin']>=word_dict['xmin'] and phone_dict['xmin']<word_dict['xmax']: #phone onset between word onset and offset
                new_phone_dict['xmin'] = phone_dict['xmin']
                new_phone_dict['xmax'] = phone_dict['xmax']
                new_phone_dict['text'] = phone_dict['text']
                new_phone_dict['in_word'].append(word_dict['text'])
                curr_in_word = word_dict['text']
                if curr_in_word != last_in_word:  # check if first phone in word and if so mark it in the corresponding field
                    new_phone_dict['first_phone'] = 1
                else:
                    new_phone_dict['first_phone'] = 0
                new_phone_dict['word_number'].append(w+1)
                
                # CHECK IF LAST PHONE
                if i_phone == len(phones) - 1: # LAST WORD
                    new_phone_dict['last_phone'] = 1
                else: # FIRST OR MIDDLE WORD
                    if phones[i_phone+1]['xmin'] >= word_dict['xmax']: # IF NEXT PHONE'S ONSET IF AFTER WORD OFFSET
                        new_phone_dict['last_phone'] = 1
                    else:
                        new_phone_dict['last_phone'] = 0
                
                phones_with_words_to_which_they_belong.append(new_phone_dict)

        assert len(new_phone_dict['in_word']) == 1
        new_phone_dict['in_word'] = new_phone_dict['in_word'][0]
        new_phone_dict['word_number'] = new_phone_dict['word_number'][0]
        last_in_word = curr_in_word

    # Add another dummy phone for END_OF_WAV
    new_phone_dict = {}
    new_phone_dict['in_word'] = '-'
    new_phone_dict['first_phone'] = -1
    new_phone_dict['last_phone'] = -1
    new_phone_dict['word_number'] = -1
    new_phone_dict['xmin'] = phones_with_words_to_which_they_belong[-1]['xmax'] # offset of last phone as END_OF_WAV
    new_phone_dict['text'] = 'END_OF_WAV'
    phones_with_words_to_which_they_belong.append(new_phone_dict)


    return phones_with_words_to_which_they_belong

def generate_new_log_lines(phones, curr_onset_time, stimulus_number):
    new_log_lines = []
    for phone_number, phone_dict in enumerate(phones):
        new_time = curr_onset_time + int(phone_dict['xmin']*1e6)
        new_line = '%i %s %i %i %i %i %i %s %s\n' % (new_time, 'PHONE_ONSET', phone_dict['first_phone'], phone_dict['last_phone'], phone_number+1, phone_dict['word_number'], stimulus_number, phone_dict['text'], phone_dict['in_word'])
        new_log_lines.append(new_line)

    return new_log_lines

#-------------------
#----- MAIN --------
#-------------------

# Generate new logs
silence_strings = ['sil', 'sp', '']
for log_fn in sorted(logs_fns):
    print('Loading %s:' % log_fn)
    with open(log_fn, 'r') as f:
        log_text = f.readlines()

    new_log = []
    for l in log_text:
        curr_line = l.strip('\n').split(' ')
        if curr_line[1] == 'AUDIO_PLAYBACK_ONSET':
            new_log.append(l)
            words, phones = get_curr_words_and_phones(curr_line, path2stimuli)
            # remove silence
            words = [w for w in words if not w['text'] in silence_strings]
            phones = [p for p in phones if not p['text'] in silence_strings]
            phones = add_words_to_which_phones_belong(words, phones)
            # generate new lines of log
            print(curr_line)
            curr_onset_time = int(curr_line[0].split('.')[0])
            stimulus_number = int(curr_line[2][0:-4])
            new_log_lines = generate_new_log_lines(phones, curr_onset_time, stimulus_number)
            new_log.extend(new_log_lines)
        else:
            new_log.append(l)

    new_log_fn = 'new_with_phones_' + os.path.basename(log_fn)
    with open(os.path.join(path2logs, new_log_fn), 'w') as f_new:
        for l in new_log:
            f_new.write(l)

    print('New log was saved to: %s' % os.path.join(path2logs, new_log_fn))
