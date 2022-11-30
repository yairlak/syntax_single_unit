import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')

from utils import load_settings_params
from scipy import io
import os, glob, argparse
import numpy as np


def HTML_per_probe(patient, comparison_name, data_type, filt, level, probe_name, path2figures='../../../Figures/', path2data='../../Data/UCLA'):
    '''
    '''
    # ENCODING
    model_type = 'ridge'
    ablation_type = 'remove'
    query_vis = 'block in [1,3,5] and word_length>1'
    query_aud = 'block in [2,4,6] and word_length>1'
    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> pt-{patient} {data_type} {probe_name} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    if level=='sentence':
        level='sentence_onset'

    if data_type=='micro':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_*_G*-{probe_name}?_[]_{comparison_name}_sentence_length_sentence_string.png'
    elif data_type=='macro':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_*_{probe_name}?_[]_{comparison_name}_sentence_length_sentence_string.png'
    elif data_type=='spike':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_*{probe_name}?_*_[]_{comparison_name}_sentence_length_sentence_string.png'
    elif data_type=='microphone':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_*{probe_name}_*_{comparison_name}_sentence_length_sentence_string.png'

    if data_type in ['micro', 'macro', 'microphone']:
        # GET FILE NAMES
        fnames = os.path.join(path2figures, 'Comparisons', comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames)
        #print(fnames)
        fnames = glob.glob(fnames)
        if len(fnames) == 0:
            print(f'No files were found for {filt}')
            
        
        # BUILD HTML 
        for fn in sorted(fnames):
            text_list.append('<br>\n')
            fn = os.path.join(path2figures[3:], 'Comparisons', comparison_name, 'patient_' + patient, 'ERPs', data_type,os.path.basename(fn))
            text_list.append(f'<img class="right" src="{fn}" stye="width:1024px;height:512px;">\n')
            fn = fn.replace('onset', 'end')
            fn = fn.replace('all_trials', 'all_end_trials')
            text_list.append(f'<img class="right" src="{fn}" stye="width:1024px;height:512px;">\n')
            # ADD ENCODING FIGURES
            parse = fn.split('_')
            #print(parse)
            if patient in ['479_11', '479_25']:
                IX = 14
            else:
                IX = 12
            electrode_name = parse[IX]
            fn_encoding = f'rf_r_patient_{patient}_{data_type}_{filt}_100_{model_type}_None_{ablation_type}_{query_vis}_False_{electrode_name}_groupped.png'
            fn_encoding = os.path.join(path2figures[3:], 'encoding_models', fn_encoding)
            text_list.append(f'<img class="right" src="{fn_encoding}" stye="width:768px;height:512px;">\n')
            fn_encoding = f'rf_r_patient_{patient}_{data_type}_{filt}_100_{model_type}_None_{ablation_type}_{query_aud}_False_{electrode_name}_groupped.png'
            fn_encoding = os.path.join(path2figures[3:], 'encoding_models', fn_encoding)
            text_list.append(f'<img class="right" src="{fn_encoding}" stye="width:768px;height:512px;">\n')
            text_list.append('<br>\n')
            
    elif data_type == 'spike':
        # GET FILENAMES OF PNG FILES
        fnames = os.path.join(path2figures, 'Comparisons', comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames)
        #print(fnames)
        fnames = glob.glob(fnames)
        num_units = len(fnames)
        if num_units == 0:
            print(f'No files were found for {fnames}')
        #print(f'Found {num_units} spike files')
        
        # BUILD HTML
        text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{probe_name}</p>')
        for fn in fnames:
            text_list.append('<br>\n')
            # Trialwise figures
            fn = os.path.join(path2figures[3:], 'Comparisons', comparison_name, 'patient_' + patient, 'ERPs', data_type,os.path.basename(fn))
            text_list.append(f'<img class="right" src="{fn}" stye="width:1024px;height:512px;">\n')
            fn = fn.replace('onset', 'end')
            fn = fn.replace('all_trials', 'all_end_trials')
            text_list.append(f'<img class="right" src="{fn}" stye="width:1024px;height:512px;">\n')
            
            # Encoding figures
            parse = fn.split('_')
            if patient in ['479_11', '479_25']:
                st, ed = 14, 16
            else:
                st, ed = 12, 14
            unit_name = '_'.join(parse[st:ed])
            #print(fn, unit_name)
            fn_encoding = f'rf_r_patient_{patient}_{data_type}_{filt}_100_{model_type}_None_{ablation_type}_{query_vis}_False_{unit_name}_groupped.png'
            fn_encoding = os.path.join(path2figures[3:], 'encoding_models', fn_encoding)
            text_list.append(f'<img class="right" src="{fn_encoding}" stye="width:768px;height:512px;">\n')
            fn_encoding = f'rf_r_patient_{patient}_{data_type}_{filt}_100_{model_type}_None_{ablation_type}_{query_aud}_False_{unit_name}_groupped.png'
            fn_encoding = os.path.join(path2figures[3:], 'encoding_models', fn_encoding)
            text_list.append(f'<img class="right" src="{fn_encoding}" stye="width:768px;height:512px;">\n')
            text_list.append('<br>\n')

            # Add Decoding Figures
    text_list.extend(add_decoding_figures(text_list, patient, data_type, filt, probe_name, path2figures))

    return text_list


def HTML_all_patients(per_probe_htmls, comparison_name, data_type, filt):
    '''
    per_probe_htmls - list of sublists; each sublist contains [patient, probe_names, fn_htmls]
    '''

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {comparison_name} {data_type} {filt} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{data_type}, {comparison_name}</p>')
    for per_probe_html in per_probe_htmls:
        patient, probe_names, fn_htmls = per_probe_html
        curr_line = f'<h2 style="font-family:tempus sans itc;"><p>{patient}:  '
        for i_probe, (probe_name, fn_html) in enumerate(zip(probe_names, fn_htmls)):
            if i_probe < len(probe_names)-1:
                space = '<a href="#" style="text-decoration:none">&nbsp;-&nbsp;</a>'
            else:
                space = ''
            curr_line += f'<a href={fn_html}>{probe_name}  </a>' + space
        text_list.append(curr_line + '</p></h2>')

    return text_list


def HTML_levels(comparison_name, data_type, filt):
    '''
    '''

    levels = ['sentence', 'word', 'phone']

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {comparison_name} {data_type} {filt} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{comparison_name}, {data_type}, {filter} </p>')
    text_list.append(f'<br>\n')
    for level in levels:
        fn_html = f'All_patients_{level}_{filt}_{data_type}_{comparison_name}.html'
        text_list.append(f'<a href={fn_html}>{level}</a>')

    return text_list


def HTML_filters(comparison_name, data_type):
    '''
    '''

    filters = ['raw', 'high-gamma']

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {comparison_name} {data_type} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{comparison_name}, {data_type} </p>')
    text_list.append(f'<br>\n')
    for filt in filters:
        fn_html = f'All_levels_{filt}_{data_type}_{comparison_name}.html'
        text_list.append(f'<a href={fn_html}>{filt}</a>')

    return text_list


def HTML_data_types(comparison_name):
    '''
    '''

    data_types = ['macro', 'micro', 'spike', 'microphone']

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {comparison_name} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{comparison_name} </p>')
    text_list.append(f'<br>\n')
    for data_type in data_types:
        fn_html = f'All_filters_{data_type}_{comparison_name}.html'
        text_list.append(f'<a href={fn_html}>{data_type}</a>')

    return text_list


def HTML_comparison_names(comparison_names):
    '''
    '''

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> Overview plots </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    # MAIN
    for comparison_name in comparison_names:
        fn_html = f'All_data_types_{comparison_name}.html'
        text_list.append(f'<a href={fn_html}>{comparison_name}</a>')
        text_list.append(f'<br>\n')

    return text_list



def add_decoding_figures(text_list, patient, data_type, filt, probe_name, path2figures):
    text_list = []
    decoding_contrasts = ['manner_of_articulation', 'word_string', 'pos_first', 'pos_simple', 'embedding_vs_long', 'wh_subj_obj_len5', 'dec_quest_len2', 'grammatical_number']
    
    # LABELS
    fn_decoding = f'label_.png'
    fn_decoding = os.path.join(path2figures[3:], 'Decoding', fn_decoding)
    text_list.append(f'<img class="right" src="{fn_decoding}" stye="width:256px;height:256px;">\n')
    for contrast in decoding_contrasts:  # Labels
        fn_decoding = f'label_{contrast}.png'
        fn_decoding = os.path.join(path2figures[3:], 'Decoding', fn_decoding)
        text_list.append(f'<img class="right" src="{fn_decoding}" stye="width:512px;height:256px;">\n')
    text_list.append('<br>\n')

    
    # DECODING PLOTS
    for block in ['V', 'A', 'V2A', 'A2V']:
        fn_decoding = f'label_{block}.png'
        fn_decoding = os.path.join(path2figures[3:], 'Decoding', fn_decoding)
        text_list.append(f'<img class="right" src="{fn_decoding}" stye="width:256px;height:256px;">\n')
        for contrast in decoding_contrasts:
            fn_decoding = get_fn_decoding(patient, data_type, filt, contrast, block, probe_name, path2figures)
            text_list.append(f'<img class="right" src="{fn_decoding}" stye="width:512px;height:512px;">\n')
        text_list.append('<br>\n')

    return text_list


def get_fn_decoding(patient, data_type, filt, contrast, block, probe_name, path2figures):
    dict_blocks = {'V':'visual', 'A':'auditory'}
    contrast_str = ''
    for b in block: # loop over characters (relevant for A2V or V2A)
        if b != '2':
            contrast_str += f'_{contrast}_{dict_blocks[b]}'
    fn = f'GAT_patient_{patient}_{data_type}_{filt}{contrast_str}_{probe_name}.png'
    fn_fullpath = os.path.join(path2figures, 'Decoding', fn)
    #print(fn)
    if os.path.isfile(fn_fullpath):
        fn = os.path.join(path2figures[3:], 'Decoding', fn) # [3:] removes '../' prefix
    else:
        fn = os.path.join(path2figures[3:], 'Decoding', 'label_blank.png')
    return fn

