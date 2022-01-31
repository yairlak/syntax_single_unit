#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:40:56 2022

@author: yl254115
"""

 str_comparison = '_'.join([tup[0] for tup in comparison['queries']])
    if not args.save2:
        if isinstance(comparison['sort'], list):
            comparison_str = '_'.join(comparison['sort'])
        else:
            comparison_str = comparison['sort']
        fname_fig = 'ERP_trialwise_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (args.patient, args.data_type, args.level, args.filter, args.smooth, ch_name, args.block_type, args.comparison_name, comparison_str)
        if args.average_repeated_trials:
            fname_fig += '_lumped'
        if args.fixed_constraint:
            fname_fig += '_'+args.fixed_constraint
        fname_fig += '.png'
        if args.responsive_channels_only:
            dname_fig = os.path.join('..', '..', 'Figures', 'Comparisons', 'responsive', args.comparison_name, args.patient, 'ERPs', args.data_type)
        else:
            dname_fig = os.path.join('..', '..', 'Figures', 'Comparisons', args.comparison_name, args.patient, 'ERPs', args.data_type)
        if not os.path.exists(dname_fig):
            os.makedirs(dname_fig)
        fname_fig = os.path.join(dname_fig, fname_fig)
    else:
        fname_fig = args.save2

    if (not os.path.exists(fname_fig)) or (not args.dont_write): # Check if output fig file already exists: 
        # Get number of trials from each query
        nums_trials = []; ims = []
        for query in comparison['queries']:
            data_curr_query = epochs[query].pick(ch_name).get_data()[:, 0, :]
            if args.average_repeated_trials:
                _, yticklabels, _ = get_sorting(epochs,
                                                query,
                                                comparison['sort'],
                                                ch_name, args)
                data_curr_query, yticklabels = average_repeated_trials(data_curr_query, yticklabels)
            nums_trials.append(data_curr_query.shape[0]) # query and pick channel
        print('Number of trials from each query:', nums_trials)
        nums_trials_cumsum = np.cumsum(nums_trials)
        nums_trials_cumsum = [0] + nums_trials_cumsum.tolist()
        # Prepare subplots
        if args.level == 'word':
            fig, _ = plt.subplots(figsize=(30, 100))
            num_queries = len(comparison['queries'])
            #height_ERP = int(np.ceil(sum(nums_trials)/num_queries))
            height_ERP = np.max(nums_trials)
        else:
            fig, _ = plt.subplots(figsize=(15, 10))
            num_queries = len(comparison['queries'])
            height_ERP = int(np.ceil(sum(nums_trials)/num_queries))
        if num_queries > 1:
            spacing = int(np.ceil(0.1*sum(nums_trials)/num_queries))
        else:
            spacing = 0
        nrows = sum(nums_trials)+height_ERP+spacing*num_queries; ncols = 10 # number of rows in subplot grid per query. Width is set to 10. num_queries is added for 1-row spacing
        # prepare axis for ERPs 
        ax2 = plt.subplot2grid((nrows, ncols+1), (sum(nums_trials)+spacing*num_queries, 0), rowspan=height_ERP, colspan=10) # Bottom figure for ERP
        # Collect data from all queries and sort based on args.sort_key
        # data = []
        evoked_dict = dict()
        colors_dict = {}
        linestyles_dict = {}
        styles = {}