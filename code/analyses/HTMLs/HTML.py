import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')

from utils import load_settings_params
from scipy import io
import os, glob, argparse
import numpy as np


def HTML_per_probe(patient, comparison_name, data_type, filt, level, probe_name, path2figures='../../../Figures/Comparisons/', path2figures_html='../../Figures/Comparisons/', path2data='../../Data/UCLA'):
    '''
    '''
    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> pt-{patient} {data_type} {probe_name} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    if data_type=='micro':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_G*-{probe_name}?_[]_{comparison_name}_sentence_length_sentence_string.png'
    elif data_type=='macro':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_{probe_name}?-{probe_name}*_[]_{comparison_name}_sentence_length_sentence_string.png'
    elif data_type=='spike':
        fnames = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_{filt}_*_{probe_name}_[]_{comparison_name}_sentence_length_sentence_string.png'

    if data_type in ['micro', 'macro']:
        # GET FILE NAMES
        fnames = os.path.join(path2figures, comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames)
        print(fnames)
        fnames = glob.glob(fnames)
        print(f'Found {len(fnames)} {filt} files')
        
        # BUILD HTML 
        for fn in sorted(fnames):
            text_list.append('<br>\n')
            fn = os.path.join(path2figures_html, comparison_name, 'patient_' + patient, 'ERPs', data_type,os.path.basename(fn))
            text_list.append(f'<img class="right" src="{fn}" stye="width:1024px;height:512px;">\n')
            text_list.append('<br>\n')
    
    elif data_type == 'spike':
        # GET FILENAMES OF PNG FILES
        fnames = os.path.join(path2figures, comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames)
        print(fnames)
        fnames = glob.glob(fnames)
        num_units = len(fnames)
        print(f'Found {num_units} spike files')
        
        # GET RESPONSE SIGNIFICANCE
        #fn_responses = os.path.join(path2data, 'patient_' + patient, 'Epochs', f'patient_{patient}_{data_type}_gaussian-kernel_{level}-epo')
        #responses_dict = {}
        #for block_type in ['Auditory', 'Visual']:
        #    responses_dict[block_type] = {}
            # READ RESPONSOVNESS FILES
        #    responses = []
        #    if len(responses) == 0:
        #        print(f'Something went wrong for: {patient} {data_type} {level} {probe_name}')
        #        return text_list
            
            # SORT BY P-VALUE
        #    channel_names = [l[0].strip() for l in responses]
        #    p_values = []
        #    for l in responses:
        #        p_vals = l[1].split(';')
        #        p_vals = [p.strip() for p in p_vals]
        #        p_vals = [float(p) for p in p_vals if p]
        #        p_values.append(p_vals)
        #    p_values_min = [min(p_vals) if p_vals else 1 for p_vals in p_values]
        #    responses_dict[block_type]['channel_names'] = channel_names
        #    responses_dict[block_type]['p_values_min'] = p_values_min
            
        #assert responses_dict['Auditory']['channel_names'] == responses_dict['Visual']['channel_names']
        #p_values_min = [min(p1, p2) for (p1, p2) in zip(responses_dict['Auditory']['p_values_min'], responses_dict['Visual']['p_values_min'])]

            #[print(c, p) for (c, p) in zip(channel_names, p_values_min)]
        #p_values_min, channe_names = zip(*sorted(zip(p_values_min, responses_dict['Auditory']['channel_names'])))

            
        # BUILD HTML
        text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{probe_name}</p>')
        #text_list.append('<br>\n')
        #for channel_name, p_val in zip(channel_names, p_values_min):
        #    fn_spike = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_raw_{channel_name}_[]_all_trials_sentence_length_sentence_string.png'
        #    fn_spike = os.path.join(path2figures_html, comparison_name, 'patient_' + patient, 'ERPs', data_type, fn_spike)
            #text_list.append(f'<p>p-value: {p_val}</p>')
        #    text_list.append(f'<img class="right" src="{fn_spike}" stye="width:1024px;height:512px;">\n')
            #text_list.append('<br>\n')

        for fn in fnames:
            text_list.append('<br>\n')
            fn = os.path.join(path2figures_html, comparison_name, 'patient_' + patient, 'ERPs', data_type,os.path.basename(fn))
            text_list.append(f'<img class="right" src="{fn}" stye="width:1024px;height:512px;">\n')
            text_list.append('<br>\n')


    return text_list


def HTML_per_probe_(patient, data_type, level, probe_name, comparison_name, path2figures='../../Figures/Comparisons/', path2figures_html='../../Figures/Comparisons/', path2data='../../Data/UCLA'):
    '''
    '''
    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> pt-{patient} {data_type} {probe_name} </title>\n')
    text_list.append('</head>\n')
    text_list.append('<body>\n')
    
    if data_type=='micro':
        fnames_kernel = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_gaussian-kernel_G*-{probe_name}?_[]_all_trials_sentence_length_sentence_string.png'
        fnames_gamma = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_high-gamma_G*-{probe_name}?_[]_all_trials_sentence_length_sentence_string.png'
    elif data_type=='macro':
        fnames_kernel = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_gaussian-kernel_{probe_name}?-{probe_name}*_[]_all_trials_sentence_length_sentence_string.png'
        fnames_gamma = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_high-gamma_{probe_name}?-{probe_name}*_[]_all_trials_sentence_length_sentence_string.png'
    elif data_type=='spike':
        fnames_spike = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_raw_*_{probe_name}_[]_all_trials_sentence_length_sentence_string.png'

    if data_type in ['micro', 'macro']:
        # GET FILE NAMES
        fnames_kernel = os.path.join(path2figures, comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames_kernel)
        fnames_gamma = os.path.join(path2figures, comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames_gamma)
        print(fnames_kernel, fnames_gamma)
        fnames_kernel = glob.glob(fnames_kernel)
        fnames_gamma = glob.glob(fnames_gamma)
        print(f'Found {len(fnames_kernel)} kernel files and {len(fnames_gamma)} high-gamma files')
        if not fnames_gamma:
            fnames_gamma = [f.replace('gaussian-kernel', 'high-gamma') for f in fnames_kernel]
        if not fnames_kernel:
            fnames_kernel = [f.replace('high-gamma', 'gaussian-kernel') for f in fnames_gamma]
        
        # BUILD HTML 
        for fn_kernel, fn_gamma in zip(sorted(fnames_kernel), sorted(fnames_gamma)):
            text_list.append('<br>\n')
            fn_kernel = os.path.join(path2figures_html, comparison_name, 'patient_' + patient, 'ERPs', data_type,os.path.basename(fn_kernel))
            fn_gamma = os.path.join(path2figures_html, comparison_name, 'patient_' + patient, 'ERPs', data_type, os.path.basename(fn_gamma))
            text_list.append(f'<img class="right" src="{fn_kernel}" stye="width:1024px;height:512px;">\n')
            text_list.append(f'<img class="right" src="{fn_gamma}" stye="width:1024px;height:512px;">\n') 
            text_list.append('<br>\n')
    
    elif data_type == 'spike':
        # GET FILENAMES OF PNG FILES
        fnames_spike = os.path.join(path2figures, comparison_name, 'patient_' + patient, 'ERPs', data_type, fnames_spike)
        print(fnames_spike)
        fnames_spike = glob.glob(fnames_spike)
        num_units = len(fnames_spike)
        print(f'Found {num_units} spike files')
        
        # GET RESPONSE SIGNIFICANCE
        fn_responses = os.path.join(path2data, 'patient_' + patient, 'Epochs', f'patient_{patient}_{data_type}_gaussian-kernel_{level}-epo')
        responses_dict = {}
        for block_type in ['Auditory', 'Visual']:
            responses_dict[block_type] = {}
            # READ RESPONSOVNESS FILES
            with open(fn_responses + '.' + block_type[:3].lower(), 'r') as f:
                responses = f.readlines()
            responses = [l.split(',')[1:3] for l in responses[4::]]
            responses = [(l[0], l[1]) for l in responses if probe_name == l[0].split('_')[-1]] # filter for current probe
            if len(responses) == 0:
                print(f'Something went wrong for: {patient} {data_type} {level} {probe_name}')
                return text_list
            
            # SORT BY P-VALUE
            channel_names = [l[0].strip() for l in responses]
            p_values = []
            for l in responses:
                p_vals = l[1].split(';')
                p_vals = [p.strip() for p in p_vals]
                p_vals = [float(p) for p in p_vals if p]
                p_values.append(p_vals)
            p_values_min = [min(p_vals) if p_vals else 1 for p_vals in p_values]
            responses_dict[block_type]['channel_names'] = channel_names
            responses_dict[block_type]['p_values_min'] = p_values_min
            
        assert responses_dict['Auditory']['channel_names'] == responses_dict['Visual']['channel_names']
        p_values_min = [min(p1, p2) for (p1, p2) in zip(responses_dict['Auditory']['p_values_min'], responses_dict['Visual']['p_values_min'])]

            #[print(c, p) for (c, p) in zip(channel_names, p_values_min)]
        p_values_min, channe_names = zip(*sorted(zip(p_values_min, responses_dict['Auditory']['channel_names'])))

            
        # BUILD HTML
        text_list.append(f'<h2 style="font-family:tempus sans itc;"><p>{probe_name}</p>')
        text_list.append('<br>\n')
        for channel_name, p_val in zip(channel_names, p_values_min):
            fn_spike = f'ERP_trialwise_patient_{patient}_{data_type}_{level}_raw_{channel_name}_[]_all_trials_sentence_length_sentence_string.png'
            fn_spike = os.path.join(path2figures_html, comparison_name, 'patient_' + patient, 'ERPs', data_type, fn_spike)
            #text_list.append(f'<p>p-value: {p_val}</p>')
            text_list.append(f'<img class="right" src="{fn_spike}" stye="width:1024px;height:512px;">\n')
            #text_list.append('<br>\n')


    return text_list


def HTML_all_patients(per_probe_htmls, comparison_name, data_type, filt, level):
    '''
    per_probe_htmls - list of sublists; each sublist contains [patient, probe_names, fn_htmls]
    '''

    text_list = []
    # HEADERS
    text_list.append('<head>\n')
    text_list.append(f'<title> {comparison_name} {data_type} {filt} {level} </title>\n')
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

    levels = ['sentence_onset', 'sentence_offset', 'word', 'phone']

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

    filters = ['raw', 'gaussian-kernel', 'gaussian-kernel-25', 'high-gamma']

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

    data_types = ['macro', 'micro', 'spike']

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

