#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:28:41 2021

@author: yl254115
"""
import pickle
from operator import itemgetter
import numpy as np


def f2(seq): 
   # order preserving
   checked = []
   for e in seq:
       if e not in checked:
           checked.append(e)
   return checked


def get_sorting(epochs, query, sort_info, ch_name, args, join_ticklabels=True):
    fields_for_sorting = []
    if  sort_info == 'clustering':
        word_strings = epochs[query].metadata['word_string']
        if '[1, 3, 5]' in query:
            block = 'block in [1,3,5]'
        elif '[2, 4, 6]' in query:
            block = 'block in [2,4,6]'
        fname = f'{args.patient}_{args.data_type}_{args.filter}_{args.smooth}_{ch_name}_{block}.clu'
        fname = f'../../Output/clustering/{fname}'
        _, _, dendro, words, _ , _= pickle.load(open(fname, 'rb'))
        index = dendro['leaves']
        word_order = np.asarray(words)[index]
        IX, yticklabels = [], []
        for target_w in word_order:
            IX_curr_word = [i for i, w in enumerate(word_strings) if w==target_w]
            IX.extend(IX_curr_word)
            yticklabels.extend([target_w]*len(IX_curr_word))
    elif isinstance(sort_info, list):
        for field in sort_info:
            fields_for_sorting.append(epochs[query].metadata[field])
        if len(fields_for_sorting) == 1:
            mylist = [(i, j) for (i, j) in zip(range(
                len(fields_for_sorting[0])), fields_for_sorting[0])]
            IX = [i[0] for i in sorted(mylist, key=itemgetter(1))]
            mylist_sorted = sorted(mylist, key=itemgetter(1))
            yticklabels = [str(e[1]) for e in mylist_sorted]
        elif len(fields_for_sorting) == 2:
            mylist = [(i, j, k) for (i, j, k) in zip(range(
                len(fields_for_sorting[0])),
                fields_for_sorting[0],
                fields_for_sorting[1])]
            IX = [i[0] for i in sorted(mylist, key=itemgetter(1, 2))]
            mylist_sorted = sorted(mylist, key=itemgetter(1, 2))
            if join_ticklabels:
                yticklabels = [str(e[1])+'-'+str(e[2]) for e in mylist_sorted]
            else:
                yticklabels = [str(e[1]) for e in mylist_sorted]
        elif len(fields_for_sorting) == 3:
            mylist = [(i, j, k, l) for (i, j, k, l) in zip(range(
                len(fields_for_sorting[0])),
                fields_for_sorting[0],
                fields_for_sorting[1],
                fields_for_sorting[2])]
            IX = [i[0] for i in sorted(mylist, key=itemgetter(1, 2, 3))]
            mylist_sorted = sorted(mylist, key=itemgetter(1, 2, 3))
            if join_ticklabels:
                yticklabels = [str(e[1])+'-'+str(e[2])+'-'+str(e[3]) for e in mylist_sorted]
            else:
                yticklabels = [str(e[1]) for e in mylist_sorted]
            
    return IX, yticklabels, fields_for_sorting




def get_sorting_IXs(metadata, cond_id, sort_info, ch_name, args, join_ticklabels=True):
    metadata = metadata[metadata['comparison']==cond_id]
    fields_for_sorting = []
    if  sort_info == 'clustering':
        word_strings = metadata['word_string']
        if '[1, 3, 5]' in query:
            block = 'block in [1,3,5]'
        elif '[2, 4, 6]' in query:
            block = 'block in [2,4,6]'
        fname = f'{args.patient}_{args.data_type}_{args.filter}_{args.smooth}_{ch_name}_{block}.clu'
        fname = f'../../Output/clustering/{fname}'
        _, _, dendro, words, _ , _= pickle.load(open(fname, 'rb'))
        index = dendro['leaves']
        word_order = np.asarray(words)[index]
        IX, yticklabels = [], []
        for target_w in word_order:
            IX_curr_word = [i for i, w in enumerate(word_strings) if w==target_w]
            IX.extend(IX_curr_word)
            yticklabels.extend([target_w]*len(IX_curr_word))
    elif isinstance(sort_info, list):
        for field in sort_info:
            fields_for_sorting.append(metadata[field])
        if len(fields_for_sorting) == 1:
            mylist = [(i, j) for (i, j) in zip(range(
                len(fields_for_sorting[0])), fields_for_sorting[0])]
            IX = [i[0] for i in sorted(mylist, key=itemgetter(1))]
            mylist_sorted = sorted(mylist, key=itemgetter(1))
            yticklabels = [str(e[1]) for e in mylist_sorted]
        elif len(fields_for_sorting) == 2:
            mylist = [(i, j, k) for (i, j, k) in zip(range(
                len(fields_for_sorting[0])),
                fields_for_sorting[0],
                fields_for_sorting[1])]
            IX = [i[0] for i in sorted(mylist, key=itemgetter(1, 2))]
            mylist_sorted = sorted(mylist, key=itemgetter(1, 2))
            if join_ticklabels:
                yticklabels = [str(e[1])+'-'+str(e[2]) for e in mylist_sorted]
            else:
                yticklabels = [str(e[1]) for e in mylist_sorted]
        elif len(fields_for_sorting) == 3:
            mylist = [(i, j, k, l) for (i, j, k, l) in zip(range(
                len(fields_for_sorting[0])),
                fields_for_sorting[0],
                fields_for_sorting[1],
                fields_for_sorting[2])]
            IX = [i[0] for i in sorted(mylist, key=itemgetter(1, 2, 3))]
            mylist_sorted = sorted(mylist, key=itemgetter(1, 2, 3))
            if join_ticklabels:
                yticklabels = [str(e[1])+'-'+str(e[2])+'-'+str(e[3]) for e in mylist_sorted]
            else:
                yticklabels = [str(e[1]) for e in mylist_sorted]
            
    return IX, yticklabels, fields_for_sorting





def average_repeated_trials(data_curr_query, yticklabels):
    unique_values = f2(yticklabels)
    data_new, yticklabels_new = [], []
    for unique_value in list(unique_values):
        IXs2value = [i for i, ll in enumerate(yticklabels) if ll==unique_value]
        data_new.append(data_curr_query[IXs2value, :].mean(axis=0))
        yticklabels_new.append(unique_value)
        
    return np.vstack(data_new), yticklabels_new