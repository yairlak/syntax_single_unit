import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# Paths
path2stimuli = os.path.join('..', '..', 'Paradigm', 'Stimuli', 'Audio', 'normalized', 'resampled_16k')


def get_curr_words_and_phones(curr_stimulus_num, path2stimuli):
    lab_fn = os.path.join(path2stimuli, str(curr_stimulus_num) + '.lab')  # contains sentence string
    with open(lab_fn, 'r') as f_lab:
        sentence = f_lab.readlines()
        sentence = sentence[0].strip('\n')
        words = [w.lower() for w in sentence.split(' ')]

    textgrid_fn = os.path.join(path2stimuli, str(curr_stimulus_num) + '.TextGrid')  # contains word and phoneme onsets
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

def update_phone_counts(phone_dict, phone_pairs_dict, curr_phones):
    '''
    update phone and phone-pair counters
    :param phone_dict: counter dict with all phones
    :param phone_pairs_dict: counter dict of dicts
    :param curr_phones: list of dicts with phone info
    :return:
    '''
    for i, curr_ph_dict in enumerate(curr_phones):
        ph = curr_ph_dict['text']
        if ph in phone_dict.keys():
            phone_dict[ph] += 1
        else:
            phone_dict[ph] = 1

        if i>0:
            if last_ph in phone_pairs_dict.keys():
                if ph in phone_pairs_dict[last_ph].keys():
                    phone_pairs_dict[last_ph][ph] += 1
                else:
                    phone_pairs_dict[last_ph][ph] = 1
            else:
                phone_pairs_dict[last_ph] = {}
                phone_pairs_dict[last_ph][ph] = 1
        last_ph = ph
    return phone_dict, phone_pairs_dict


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


def find_phone_class(phone):
    manner_classes = ['STOP', 'AFFR', 'FRIC', 'NAS', 'LIQ', 'SIB', 'GLI']
    vowels, consonants = get_phone_classes()
    for phone_class, phones_str in vowels.items():
        if phone in phones_str.split(' '):
            return 'VOWEL'

    for phone_class, phones_str in consonants.items():
        if phone_class in manner_classes:
            if phone in phones_str.split(' '):
                return phone_class

    return 'SILENCE'

#-------------------
#----- MAIN --------
#-------------------
phone_dict = {}
phone_pairs_dict = {}
silence_strings = ['sil', 'sp', '']
for stimulus in range(1, 153):
    words, phones = get_curr_words_and_phones(stimulus, path2stimuli)
    phone_dict, phone_pairs_dict = update_phone_counts(phone_dict, phone_pairs_dict, phones)


#######
df = pd.DataFrame(columns=['phone','class','count'])
for i, (ph, count) in enumerate(phone_dict.items()):
    df.loc[i] = [ph, find_phone_class(ph), count]

df = df.sort_values(by=['count', 'class'], ascending=[False, True])
df = df[df['class'] != 'SILENCE']
print(df)

num_phones = df['count'].sum()
print(f'Total number of phones {num_phones}')

fig_barplot, ax = plt.subplots(figsize=(20,10))
ax = sns.barplot(x='class', y='count', hue='phone', data=df)
ax.legend(bbox_to_anchor=(1, 1), fontsize=8)
plt.savefig('../../Figures/phone_dist.png')
plt.close(fig_barplot)


fig_heatmap, ax = plt.subplots(figsize=(15,15))
df = pd.DataFrame(columns=['pre_phone','post_phone', 'class_pre', 'class_post', 'count'])
cnt = 0
for pre_ph, post_phs in phone_pairs_dict.items():
    for post_ph, count in post_phs.items():
        df.loc[cnt] = [pre_ph, post_ph, find_phone_class(pre_ph), find_phone_class(post_ph), int(count)]
        cnt+=1


df = df.sort_values(by=['count', 'class_pre', 'class_post'], ascending=[False, True, True])
df = df[df['class_pre'] != 'SILENCE']
df = df[df['class_post'] != 'SILENCE']
print(df[df['count']>6])
piv = df.pivot_table(index=['pre_phone'], columns=['post_phone'], values='count', fill_value=0, aggfunc=np.sum)
#print(piv)
ax = sns.heatmap(piv, square=True, annot=True)
# ax.get_legend().remove()
plt.tight_layout()
plt.savefig('../../Figures/phone_transition.png')
plt.close()

