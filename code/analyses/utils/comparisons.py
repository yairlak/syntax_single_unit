def comparison_list():
    comparisons = {}
# 'high_low_op_nodes', 'morph_complex', 'grammatical_number', 'gender', 'subject_type', 'high_low_freq', 'pos_simple', 'tense', 'manner_of_articulation', 'word_string'

    vowels, consonants = get_phone_classes()
# FOR HTMLs
    
    semantic_categories = ['abstract', 'action', 'body', 'emotion', 'event',
                           'flower', 'food', 'fun', 'mental', 'movement',
                           'music', 'negative', 'object', 'perception',
                           'person', 'question', 'relation', 'search',
                           'sleep', 'speech', 'vehicle', 'water']
    comparisons['semantic_categories_vis'] = {}
    comparisons['semantic_categories_vis']['queries'] = [f"semantic_categories_names.str.contains('{c}') and (block in [1, 3, 5])" for c in semantic_categories]
    comparisons['semantic_categories_vis']['condition_names'] = [f'{c}' for c in semantic_categories]
    comparisons['semantic_categories_vis']['colors'] = []
    comparisons['semantic_categories_vis']['cmaps'] = ['Reds' for _ in semantic_categories]
    comparisons['semantic_categories_vis']['sort'] = ['word_string']
    comparisons['semantic_categories_vis']['level'] = 'word'
    comparisons['semantic_categories_vis']['tmin_tmax'] = [-0.25, 0.8]

    comparisons['semantic_categories_aud'] = {}
    comparisons['semantic_categories_aud']['queries'] = [f"semantic_categories_names.str.contains('{c}') and (block in [2, 4, 6])" for c in semantic_categories]
    comparisons['semantic_categories_aud']['condition_names'] = [f'{c}' for c in semantic_categories]
    comparisons['semantic_categories_aud']['colors'] = []
    comparisons['semantic_categories_aud']['cmaps'] = ['Reds' for _ in semantic_categories]
    comparisons['semantic_categories_aud']['sort'] = ['word_string']
    comparisons['semantic_categories_aud']['level'] = 'word'
    comparisons['semantic_categories_aud']['tmin_tmax'] = [-0.25, 0.8]

    # SYNTAX
    comparisons['embedding_vs_long'] = {}
    comparisons['embedding_vs_long']['queries'] = ["embedding==True",
                                                   "dec_quest==0 and sentence_length==5 and embedding==0"]
    comparisons['embedding_vs_long']['condition_names'] = ['Embedded', 'Long']
    comparisons['embedding_vs_long']['colors'] = ['b', 'r']
    comparisons['embedding_vs_long']['ls'] = ['--', '-']
    comparisons['embedding_vs_long']['level'] = 'sentence_offset'
    comparisons['embedding_vs_long']['tmin_tmax'] = [-3.5, 1]
    
    #comparisons['embedding_vs_long'] = {}
    #comparisons['embedding_vs_long']['queries'] = ["embedding==True and block_type=='visual'",
    #                                               "dec_quest==0 and sentence_length==5 and embedding==0 and block_type=='visual'",
    #                                               "embedding==True and block_type=='auditory'",
    #                                               "dec_quest==0 and sentence_length==5 and embedding==0 and block_type=='auditory'"]
    #comparisons['embedding_vs_long']['condition_names'] = ['Embedded Visual', 'Long Visual', 'Embedded Auditory', 'Long Auditory']
    #comparisons['embedding_vs_long']['colors'] = ['b', 'b', 'r', 'r']
    #comparisons['embedding_vs_long']['ls'] = ['--', '-', '--', '-']
    #comparisons['embedding_vs_long']['level'] = 'sentence_offset'
    #comparisons['embedding_vs_long']['tmin_tmax'] = [-3.5, 1]
    
    comparisons['wh_subj_obj_len5'] = {}
    comparisons['wh_subj_obj_len5']['queries'] = ["wh_subj_obj==1 and sentence_length>3", "wh_subj_obj==-1 and sentence_length>3"]
    comparisons['wh_subj_obj_len5']['condition_names'] = ['Subject question', 'Object question']
    comparisons['wh_subj_obj_len5']['colors'] = ['b', 'g']
    comparisons['wh_subj_obj_len5']['level'] = 'sentence_onset'
    comparisons['wh_subj_obj_len5']['tmin_tmax'] = [-0.5, 3.5]
    
    comparisons['dec_quest_len2'] = {}
    comparisons['dec_quest_len2']['queries'] = ["dec_quest==0 and sentence_length==2", "dec_quest==1 and sentence_length==2"]
    comparisons['dec_quest_len2']['condition_names'] = ['Declarative', 'Question']
    comparisons['dec_quest_len2']['colors'] = ['b', 'g']
    comparisons['dec_quest_len2']['sort'] = ['word_string']
    comparisons['dec_quest_len2']['level'] = 'sentence_onset'
    comparisons['dec_quest_len2']['tmin_tmax'] = [-0.5, 2.5]



    comparisons['dec_quest_len2_vis_aud'] = {}
    comparisons['dec_quest_len2_vis_aud']['queries'] = ["dec_quest==0 and sentence_length==2 and (block in [1, 3, 5])",
                                                "dec_quest==1 and sentence_length==2 and (block in [1, 3, 5])",
                                                "dec_quest==0 and sentence_length==2 and (block in [2, 4, 6])",
                                                "dec_quest==1 and sentence_length==2 and (block in [2, 4, 6])"]
    comparisons['dec_quest_len2_vis_aud']['condition_names'] = ['Declarative-visual',
                                                                'Question-visual',
                                                                'Declarative-auditory',
                                                                'Question-auditory']
    comparisons['dec_quest_len2_vis_aud']['colors'] = ['r', 'r', 'b', 'b']
    comparisons['dec_quest_len2_vis_aud']['ls'] = ['-', '--', '-', '--']
    comparisons['dec_quest_len2_vis_aud']['cmaps'] = ['Reds', 'Reds', 'Blues', 'Blues']
    comparisons['dec_quest_len2_vis_aud']['sort'] = 'rate'
    comparisons['dec_quest_len2_vis_aud']['level'] = 'sentence_onset'
    comparisons['dec_quest_len2_vis_aud']['tmin_tmax'] = [-0.25, 1.5]
    comparisons['dec_quest_len2_vis_aud']['ylim'] = 35
    
    comparisons['dec_quest_len2_vis'] = {}
    comparisons['dec_quest_len2_vis']['queries'] = ["dec_quest==0 and sentence_length==2 and (block in [1, 3, 5])",
                                                "dec_quest==1 and sentence_length==2 and (block in [1, 3, 5])"]
    comparisons['dec_quest_len2_vis']['condition_names'] = ['Declarative-visual',
                                                                'Question-visual']
    comparisons['dec_quest_len2_vis']['colors'] = ['r', 'b']
    comparisons['dec_quest_len2_vis']['ls'] = ['-', '--']
    comparisons['dec_quest_len2_vis']['cmaps'] = ['Reds', 'Reds']
    comparisons['dec_quest_len2_vis']['sort'] = 'rate'
    comparisons['dec_quest_len2_vis']['level'] = 'sentence_onset'
    comparisons['dec_quest_len2_vis']['tmin_tmax'] = [-0.25, 1.5]
    comparisons['dec_quest_len2_vis']['ylim'] = 35
    
    comparisons['dec_quest_len2_aud'] = {}
    comparisons['dec_quest_len2_aud']['queries'] = ["dec_quest==0 and sentence_length==2 and (block in [2, 4, 6])",
                                                "dec_quest==1 and sentence_length==2 and (block in [2, 4, 6])"]
    comparisons['dec_quest_len2_aud']['condition_names'] = ['Declarative-auditory',
                                                                'Question-auditory']
    comparisons['dec_quest_len2_aud']['colors'] = ['r', 'b']
    comparisons['dec_quest_len2_aud']['ls'] = ['-', '--']
    comparisons['dec_quest_len2_aud']['cmaps'] = ['Blues', 'Blues']
    comparisons['dec_quest_len2_aud']['sort'] = 'rate'
    comparisons['dec_quest_len2_aud']['level'] = 'sentence_onset'
    comparisons['dec_quest_len2_aud']['tmin_tmax'] = [-0.25, 1.5]
    comparisons['dec_quest_len2_aud']['ylim'] = 35
    
    
    comparisons['grammatical_number'] = {}
    comparisons['grammatical_number']['queries'] = ["grammatical_number==-1 and block_type=='visual' and pos_simple=='PRP'",
                                                    "grammatical_number==1 and block_type=='visual' and pos_simple=='PRP'",
                                                    "grammatical_number==-1 and block_type=='auditory' and pos_simple=='PRP'",
                                                    "grammatical_number==1 and block_type=='auditory' and pos_simple=='PRP'",
                                                    "grammatical_number==-1 and block_type=='visual' and pos_simple=='VB'",
                                                    "grammatical_number==1 and block_type=='visual' and pos_simple=='VB'",
                                                    "grammatical_number==-1 and block_type=='auditory' and pos_simple=='VB'",
                                                    "grammatical_number==1 and block_type=='auditory' and pos_simple=='VB'",
                                                    "grammatical_number==-1 and block_type=='visual' and pos_simple=='NN'",
                                                    "grammatical_number==1 and block_type=='visual' and pos_simple=='NN'",
                                                    "grammatical_number==-1 and block_type=='auditory' and pos_simple=='NN'",
                                                    "grammatical_number==1 and block_type=='auditory' and pos_simple=='NN'"]
    comparisons['grammatical_number']['condition_names'] = ['Singular Visual PRP', 'Plural Visual PRP', 'Singular Auditory PRP', 'Plural Auditory PRP',
                                                           'Singular Visual VB', 'Plural Visual VB', 'Singular Auditory VB', 'Plural Auditory VB',
                                                           'Singular Visual NN', 'Plural Visual NN', 'Singular Auditory NN', 'Plural Auditory NN']
    comparisons['grammatical_number']['colors'] = ['b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r']
    comparisons['grammatical_number']['ls'] = ['--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-']
    comparisons['grammatical_number']['level'] = 'word'
    comparisons['grammatical_number']['y-tick-step'] = 20
    comparisons['grammatical_number']['sort'] = ['word_string']
    
    comparisons['A_movement'] = {}
    comparisons['A_movement']['queries'] = ["different_thematic_role==0 and sentence_length==2 and block_type=='visual'",
                                            "different_thematic_role==1 and sentence_length==2 and block_type=='visual'",
                                            "different_thematic_role==0 and sentence_length==2 and block_type=='auditory'",
                                            "different_thematic_role==1 and sentence_length==2 and block_type=='auditory'"]
    comparisons['A_movement']['condition_names'] = ['Without A-movement', 'With A-movement']
    comparisons['A_movement']['colors'] = ['b', 'g']
    comparisons['A_movement']['level'] = 'sentence'
    comparisons['A_movement']['y-tick-step'] = 20
    comparisons['A_movement']['sort'] = ['sentence_string']
    
    
    # Examples
    target_words = [['The', 'the'], ['They'], ['that'], ['What', 'Who', 'Whom']]
    comparisons['479_11_LSTG7_15p2'] = {}
    comparisons['479_11_LSTG7_15p2']['queries'] = [f"word_string in {l} and block in [2, 4, 6]" for l in target_words]
    n_queries = len(comparisons['479_11_LSTG7_15p2']['queries'])
    comparisons['479_11_LSTG7_15p2']['condition_names'] = ['The|the', 'They', 'that', 'What|Who|Whom']
    comparisons['479_11_LSTG7_15p2']['colors'] = ['g', 'r', 'm', 'b']
    comparisons['479_11_LSTG7_15p2']['cmaps'] = ['Blues'] * n_queries
    comparisons['479_11_LSTG7_15p2']['ls'] = ['-'] * n_queries
    comparisons['479_11_LSTG7_15p2']['level'] = 'word'
    comparisons['479_11_LSTG7_15p2']['figsize'] = (4, 15)
    comparisons['479_11_LSTG7_15p2']['y-tick-step'] = [50] * n_queries
    comparisons['479_11_LSTG7_15p2']['ylim'] = 50   
    comparisons['479_11_LSTG7_15p2']['sort'] = 'rate'#['word_string']
    comparisons['479_11_LSTG7_15p2']['tmin_tmax'] = [-0.05, 0.4]
    comparisons['479_11_LSTG7_15p2']['height_ratios'] = True
    comparisons['479_11_LSTG7_15p2']['channel_name'] = 'p_g2_15_GA2-LST'
    
    target_words = ['He', 'Whom', 'Who', 'They', 'We', 'She', 'What', 'The']
    comparisons['479_11_RPSTG_46p1'] = {}
    comparisons['479_11_RPSTG_46p1']['queries'] = [f"word_string=='{w}' and block in [2, 4, 6]" for w in target_words] \
                                                  + [f"word_string not in {target_words} and (block in [2, 4, 6])"]
    n_queries = len(comparisons['479_11_RPSTG_46p1']['queries'])
    comparisons['479_11_RPSTG_46p1']['condition_names'] = target_words + ['Non-inital word']
    comparisons['479_11_RPSTG_46p1']['colors'] = None
    comparisons['479_11_RPSTG_46p1']['cmaps'] = ['Blues'] * n_queries
    comparisons['479_11_RPSTG_46p1']['ls'] = ['-'] * n_queries
    comparisons['479_11_RPSTG_46p1']['level'] = 'word'
    comparisons['479_11_RPSTG_46p1']['y-tick-step'] = [50] * (n_queries-1) + [300]
    comparisons['479_11_RPSTG_46p1']['ylim'] = 30
    comparisons['479_11_RPSTG_46p1']['sort'] = 'rate'#['word_string']
    comparisons['479_11_RPSTG_46p1']['tmin_tmax'] = [-0.05, 0.4]
    comparisons['479_11_RPSTG_46p1']['height_ratios'] = True
    comparisons['479_11_RPSTG_46p1']['channel_name'] = 'p_g1_46_GB2-RPS'
    
    

    comparisons['505_LFGP6_30p2'] = {}
    comparisons['505_LFGP6_30p2']['queries'] = ["word_string.str.contains('w') and ~word_string.str.startswith('W') and ~word_string.str.startswith('w')",
                                                 "word_string.str.startswith('W') or word_string.str.startswith('w')",
                                                 "(word_string.str.contains('m') or word_string.str.contains('v')) and (~word_string.str.contains('W') and ~word_string.str.contains('w'))"]
    #comparisons['505_LFGP6_30p2']['queries'] = ["word_string.str.contains('w') and ~word_string.str.startswith('W') and ~word_string.str.startswith('w')",
                                                #"word_string.str.startswith('W') or word_string.str.startswith('w')"]
                                                #"(word_string.str.contains('m') or word_string.str.contains('v')) and (~word_string.str.contains('W') and ~word_string.str.contains('w'))"]
    
    comparisons['505_LFGP6_30p2']['condition_names'] = ['Non-initial w', 'Initital W', 'Without w']
    comparisons['505_LFGP6_30p2']['colors'] = ['g', 'r', 'k']
    comparisons['505_LFGP6_30p2']['ls'] = ['-', '-', '-']
    comparisons['505_LFGP6_30p2']['lw'] = [3, 3, 3]
    comparisons['505_LFGP6_30p2']['level'] = 'word'
    comparisons['505_LFGP6_30p2']['y-tick-step'] = [50] * (n_queries)
    comparisons['505_LFGP6_30p2']['sort'] = ['word_string']
    comparisons['505_LFGP6_30p2']['tmin_tmax'] = [-0.1, 0.6]
    comparisons['505_LFGP6_30p2']['figsize'] = (4, 15)
    comparisons['505_LFGP6_30p2']['ylim'] = 30
    comparisons['505_LFGP6_30p2']['cmaps'] = ['Reds'] * len(comparisons['505_LFGP6_30p2']['queries'])
    comparisons['505_LFGP6_30p2']['height_ratios'] = True
    comparisons['505_LFGP6_30p2']['channel_name'] = 'p_g2_30_GA4-LFG'
   
    target_words = [['She', 'she'], ['showered', 'shaved', 'chef'], ['mushrooms', 'pushed', 'vanish', 'washed']]
    control_words = [['smiles', 'smile', 'smiled', 'sing', 'sneezes', 'sneeze', 'sneezed']]
    comparisons['505_LHSG_17p1'] = {}
    comparisons['505_LHSG_17p1']['queries'] = [f"word_string in {l} and (block in [2, 4, 6])" for l in target_words] \
                                                + [f"word_string in {l} and (block in [2, 4, 6])" for l in control_words]
    n_queries = len(comparisons['505_LHSG_17p1']['queries'])
    comparisons['505_LHSG_17p1']['condition_names'] = ['Initial /S/ (she)', 'Initial /S/ (showered|shaved|chef)', 'Non-initial /S/', 'Other stridents']
    #['|'.join(l) for l in target_words] + ['|'.join(l) for l in control_words]
    comparisons['505_LHSG_17p1']['ls'] = ['-'] * n_queries
    comparisons['505_LHSG_17p1']['lw'] = [3] * n_queries
    comparisons['505_LHSG_17p1']['level'] = 'word'
    comparisons['505_LHSG_17p1']['y-tick-step'] = [300, 10, 10, 10]
    comparisons['505_LHSG_17p1']['sort'] = 'rate'
    comparisons['505_LHSG_17p1']['tmin_tmax'] = [-0.05, 0.4]
    comparisons['505_LHSG_17p1']['figsize'] = (4, 15)
    comparisons['505_LHSG_17p1']['ylim'] = 200
    comparisons['505_LHSG_17p1']['cmaps'] = ['Blues'] * n_queries
    comparisons['505_LHSG_17p1']['colors'] = ['g', 'g', 'b', 'r']
    comparisons['505_LHSG_17p1']['height_ratios'] = True
    comparisons['505_LHSG_17p1']['channel_name'] = 'p_g1_17_GA3-LHS'
   
    # ORTHOGRAPHY
    comparisons['the_they'] = {}
    comparisons['the_they']['queries'] = ["word_position==1 and word_string in ['The']", "word_position==1 and word_string in ['They']"]
    comparisons['the_they']['condition_names'] = ["The", "They"]
    comparisons['the_they']['colors'] = ["b", "r"]
    comparisons['the_they']['sort'] = ['word_string']
    comparisons['the_they']['level'] = 'sentence_onset'
    comparisons['the_they']['tmin_tmax'] = [-0.2, 0.6]
    
    comparisons['he_she'] = {}
    comparisons['he_she']['queries'] = ["word_position==1 and word_string in ['He']", "word_position==1 and word_string in ['She']"]
    comparisons['he_she']['condition_names'] = ["He", "She"]
    comparisons['he_she']['colors'] = ["b", "r"]
    comparisons['he_she']['sort'] = ['word_string']
    comparisons['he_she']['level'] = 'sentence_onset'
    comparisons['he_she']['tmin_tmax'] = [-0.2, 0.6]
    
    comparisons['what_who'] = {}
    comparisons['what_who']['queries'] = ["word_position==1 and word_string in ['What']", "word_position==1 and word_string in ['Who']"]
    comparisons['what_who']['condition_names'] = ["What", "Who"]
    comparisons['what_who']['colors'] = ["b", "r"]
    comparisons['what_who']['sort'] = ['word_string']
    comparisons['what_who']['level'] = 'sentence_onset'
    comparisons['what_who']['tmin_tmax'] = [-0.2, 0.6]


    # LEXICON
    comparisons['pos_simple'] = {}
    comparisons['pos_simple']['queries'] = "pos_simple"
    comparisons['pos_simple']['condition_names'] = []
    comparisons['pos_simple']['colors'] = []
    comparisons['pos_simple']['level'] = 'word'
    comparisons['pos_simple']['tmin_tmax'] = [-0.5, 1.2]
    
    comparisons['tense'] = {}
    comparisons['tense']['queries'] = "tense"
    comparisons['tense']['condition_names'] = []
    comparisons['tense']['colors'] = []
    comparisons['tense']['level'] = 'sentence_offset'
    
    comparisons['pos_first'] = {}
    comparisons['pos_first']['queries'] = ["pos=='DT'", "pos=='WP'", "pos=='PRP'"]
    comparisons['pos_first']['condition_names'] = ["Determiner", "WH-question", "Pronoun"]
    comparisons['pos_first']['colors'] = ["b", "r", "g"]
    comparisons['pos_first']['sort'] = ['word_string']
    comparisons['pos_first']['level'] = 'sentence_onset'
    comparisons['pos_first']['tmin_tmax'] = [-0.5, 1.2]

    comparisons['word_zipf'] = {}
    comparisons['word_zipf']['queries'] = ["(word_zipf>=%i) and (word_zipf<%i) and (block in [1, 3, 5])"%(i, j) for (i, j) in zip(range(3, 7), range(4, 7))]
    comparisons['word_zipf']['condition_names'] = [str(i-1)+'<zipf<'+str(i) for i in range(4, 7)]
    comparisons['word_zipf']['colors'] = []
    comparisons['word_zipf']['sort'] = ['word_string']
    comparisons['word_zipf']['level'] = 'word'

    # SEQUENTIAL
    comparisons['word_position'] = {}
    comparisons['word_position']['queries'] = ["word_position==1 or word_position==2", "word_position==4 or word_position==5"]
    comparisons['word_position']['condition_names'] = ["First-second", "Fourth-fifth"]
    comparisons['word_position']['colors'] = ["b", "r"]
    comparisons['word_position']['sort'] = ['word_string']
    comparisons['word_position']['level'] = 'word'

    # ORTHOGRAPHY
    comparisons['word_string_first'] = {}
    comparisons['word_string_first']['queries'] = "word_string"
    comparisons['word_string_first']['fixed_constraint'] = "word_position==1"
    comparisons['word_string_first']['condition_names'] = []
    comparisons['word_string_first']['colors'] = []
    comparisons['word_string_first']['level'] = 'word'
    comparisons['word_string_first']['y-tick-step'] = 20
    comparisons['word_string_first']['tmin_tmax'] = [-0.5, 1.2]
    comparisons['word_string_first']['sort'] = ['block_type', 'chronological_order']
    
    comparisons['word_length'] = {}
    #comparisons['word_length']['queries'] = ["word_length>1 and word_length<4", "word_length>5"]
    comparisons['word_length']['queries'] = 'word_length'
    comparisons['word_length']['condition_names'] = []
    comparisons['word_length']['colors'] = ['b', 'r']
    comparisons['word_length']['level'] = 'word'
    comparisons['word_length']['y-tick-step'] = 20
    comparisons['word_length']['tmin_tmax'] = [-0.5, 1.2]
    comparisons['word_length']['sort'] = ['block_type', 'chronological_order']
    
    comparisons['word_string_visual'] = {}
    comparisons['word_string_visual']['queries'] = "word_string"
    comparisons['word_string_visual']['fixed_constraint'] = "block_type=='visual'"
    comparisons['word_string_visual']['condition_names'] = []
    comparisons['word_string_visual']['colors'] = []
    comparisons['word_string_visual']['level'] = 'word'
    comparisons['word_string_visual']['y-tick-step'] = 20
    comparisons['word_string_visual']['tmin_tmax'] = [-0.5, 1.2]
    comparisons['word_string_visual']['sort'] = ['chronological_order']

    comparisons['word_string_auditory'] = {}
    comparisons['word_string_auditory']['queries'] = "word_string"
    comparisons['word_string_auditory']['fixed_constraint'] = "block_type=='auditory'"
    comparisons['word_string_auditory']['condition_names'] = []
    comparisons['word_string_auditory']['colors'] = []
    comparisons['word_string_auditory']['level'] = 'word'
    comparisons['word_string_auditory']['y-tick-step'] = 20
    comparisons['word_string_auditory']['tmin_tmax'] = [-0.25, 0.6]
    #comparisons['word_string_auditory']['sort'] = []

    # PHONOLOGICAL
    comparisons['manner_of_articulation'] = {}
    comparisons['manner_of_articulation']['queries'] = []
    comparisons['manner_of_articulation']['condition_names'] = []
    for cl in ['STOP', 'FRIC', 'NAS', 'LIQ', 'SIB', 'GLI']:
        curr_class = consonants[cl].split(' ')
        #curr_class = ['\\"' + s.strip(' ') + '\\"' for s in curr_class]
        curr_class = ['"'+s.strip(' ')+'"' for s in curr_class]
        curr_query = "phone_string in [%s]" % (', '.join(curr_class))
        comparisons['manner_of_articulation']['queries'].append(curr_query)
        comparisons['manner_of_articulation']['condition_names'].append(cl)
    comparisons['manner_of_articulation']['colors'] = ['r', 'g', 'b', 'y', 'c', 'm']
    comparisons['manner_of_articulation']['sort'] = ['phone_string', 'phone_position']
    comparisons['manner_of_articulation']['tmin_tmax'] = [-0.1, 0.4]
    comparisons['manner_of_articulation']['y-tick-step'] = 60
    comparisons['manner_of_articulation']['level'] = 'phone'


# ALL TRIALS
    comparisons['all_end_trials'] = {}
    comparisons['all_end_trials']['queries'] = ["is_last_word==1 and (block in [1, 3, 5])", "is_last_word==1 and (block in [2, 4, 6])"]
    comparisons['all_end_trials']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_end_trials']['colors'] = ['r', 'b']
    comparisons['all_end_trials']['cmaps'] = ['Reds', 'Blues']
    comparisons['all_end_trials']['sort'] = 'rate'#['sentence_length', 'sentence_string']
    comparisons['all_end_trials']['y-tick-step'] = [50, 50]
    comparisons['all_end_trials']['tmin_tmax'] = [-1.25, 0.5]
    comparisons['all_end_trials']['level'] = 'sentence_offset'
    comparisons['all_end_trials']['ylim'] = 40
    comparisons['all_end_trials']['figsize'] = (10, 10)

# ALL TRIALS
    comparisons['all_trials'] = {}
    comparisons['all_trials']['queries'] = ["word_position==1 and (block in [1, 3, 5])", "word_position==1 and (block in [2, 4, 6])"]
    comparisons['all_trials']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_trials']['colors'] = ['b', 'r']
    comparisons['all_trials']['sort'] = ['sentence_length', 'sentence_string']#, 'Question']
    comparisons['all_trials']['cmaps'] = ['Reds', 'Blues']
    #comparisons['all_trials']['sort'] = 'rate'
    comparisons['all_trials']['y-tick-step'] = [50, 50]
    #comparisons['all_trials']['sort'] = ['chronological_order']#, 'Question']
    comparisons['all_trials']['tmin_tmax'] = [-0.25, 1.5]
    comparisons['all_trials']['figsize'] = (10, 10)
    comparisons['all_trials']['ylim'] = 30

# ALL TRIALS CHRONOLOGICAL ORDER
    comparisons['all_trials_chrono'] = {}
    comparisons['all_trials_chrono']['queries'] = ["word_position==1 and (block in [1, 3, 5])", "word_position==1 and (block in [2, 4, 6])"]
    comparisons['all_trials_chrono']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_trials_chrono']['colors'] = ['b', 'r']
    comparisons['all_trials_chrono']['cmaps'] = ['Blues', 'Reds']
    #comparisons['all_trials']['sort'] = ['sentence_length', 'sentence_string']#, 'Question']
    comparisons['all_trials_chrono']['sort'] = ['chronological_order']#, 'Question']
    comparisons['all_trials_chrono']['tmin_tmax'] = [-0.25, 2.75]

# ALL WORDS
    comparisons['all_words'] = {}
    comparisons['all_words']['queries'] = ["word_string.str.len()>1 and (block in [1, 3, 5])", "word_string.str.len()>1 and (block in [2, 4, 6])"]
    comparisons['all_words']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_words']['colors'] = ['b', 'r']
    comparisons['all_words']['cmaps'] = ['Blues', 'Reds']
    #comparisons['all_words']['sort'] = ['word_string']
    #comparisons['all_words']['sort'] = ['word_length', 'word_string']
    comparisons['all_words']['sort'] = 'rate'
    comparisons['all_words']['y-tick-step'] = 40
    comparisons['all_words']['level'] = 'word'
    comparisons['all_words']['tmin_tmax'] = [-0.3, 0.6]



# ALL WORDS
    comparisons['food_related'] = {}
    comparisons['food_related']['queries'] = ["word_string.str.len()>1 and (block in [1, 3, 5]) and word_string.str.contains('chef|salad|mushrooms|ate|grapes|cake|sandwich|eat|meal|tasty|eats|drink|cook|overcooked')",
                                              "word_string.str.len()>1 and (block in [2, 4, 6]) and ~word_string.str.contains('chef|salad|mushrooms|ate|grapes|cake|sandwich|eat|meal|tasty|eats|drink|cook|overcooked')"]
    comparisons['food_related']['condition_names'] = ['Food related', 'Food unrelated']
    comparisons['food_related']['colors'] = ['b', 'r']
    comparisons['food_related']['sort'] = ['word_string']
    comparisons['food_related']['y-tick-step'] = 20
    comparisons['food_related']['level'] = 'word'
    comparisons['food_related']['tmin_tmax'] = [-0.3, 0.6]

# ALL PHONES
    comparisons['all_phones'] = {}
    comparisons['all_phones']['queries'] = ["phone_position>0 and (block in [1, 3, 5])", "phone_position>0 and (block in [2, 4, 6])"]
    comparisons['all_phones']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_phones']['colors'] = ['b', 'r']
    comparisons['all_phones']['sort'] = ['phone_string']
    # comparisons['all_phones']['sort'] = 'clustering'
    comparisons['all_phones']['y-tick-step'] = 1

# ALL PHONES
    comparisons['all_phones'] = {}
    comparisons['all_phones']['queries'] = 'phone_string'
    comparisons['all_phones']['condition_names'] = ['Visual blocks', 'Auditory blocks']
    comparisons['all_phones']['fixed_constraint'] = "phone_string>0"
    comparisons['all_phones']['colors'] = ['b', 'r']
    comparisons['all_phones']['sort'] = ['phone_string']
    # comparisons['all_phones']['sort'] = 'clustering'
    comparisons['all_phones']['y-tick-step'] = 1
    
# ALL WORDS
    comparisons['all_words_visual'] = {}
    comparisons['all_words_visual']['queries'] = ["word_string.str.len()>1 and (block in [1, 3, 5])"]
    comparisons['all_words_visual']['condition_names'] = ['all_words_visual']
    comparisons['all_words_visual']['colors'] = ['r']
    comparisons['all_words_visual']['cmaps'] = ['Reds']
    comparisons['all_words_visual']['sort'] = 'rate'#['word_string']
    # comparisons['all_words_visual']['sort'] = 'clustering'
    comparisons['all_words_visual']['tmin_tmax'] = [-0.1, 0.6]
    comparisons['all_words_visual']['y-tick-step'] = 20
    comparisons['all_words_visual']['level'] = 'word'

    comparisons['all_words_audio'] = {}
    comparisons['all_words_audio']['level'] = 'word'
    comparisons['all_words_audio']['queries'] = ["word_string.str.len()>1 and (block in [2, 4, 6])"]
    comparisons['all_words_audio']['condition_names'] = ['all_words_audio']
    comparisons['all_words_audio']['colors'] = ['b']
    comparisons['all_words_audio']['cmaps'] = ['Blues']
    comparisons['all_words_audio']['sort'] = ['word_string']
    #comparisons['all_words_audio']['sort'] = 'rate'
    comparisons['all_words_audio']['tmin_tmax'] = [-0.1, 0.6]
    comparisons['all_words_audio']['y-tick-step'] = 20

# Sanity checks:
    comparisons['first_last_word'] = {}
    comparisons['first_last_word']['queries'] = ["word_position==1", "last_word==True"]
    comparisons['first_last_word']['condition_names'] = ['First_word', 'Last_word']
    comparisons['first_last_word']['colors'] = ['b', 'g']




# GRAMMATICAL NUMBER:
    # Nouns:
    #
    comparisons['test'] = {}
    comparisons['test']['queries'] = ['sentence_length == 2 and last_word==1', 'sentence_length == 3 and last_word==1', 'sentence_length == 4 and last_word==1', 'sentence_length ==5 and last_word==1']
    comparisons['test']['condition_names'] = ['2', '3', '4', '5']
    comparisons['test']['colors'] = ['k', 'r', 'g', 'b']
    comparisons['test']['ls'] = ['-', '-', '-', '-']
    comparisons['test']['sort'] = ['word_string']
    # Nouns and verbs
    comparisons['number_nouns_verbs'] = {}
    comparisons['number_nouns_verbs']['queries'] = ["word_position==2 and pos in ['NN']", 
                                                             "word_position==2 and pos in ['NNs'])", 
                                                             "word_position==2 and pos in ['VBZ']",
                                                             "word_position==2 and pos in ['VB']"]
    comparisons['number_nouns_verbs']['condition_names'] = ['Noun Singular', 'Noun Plural', 'Verb Singular', 'Verb Plural']
    comparisons['number_nouns_verbs']['colors'] = ['r', 'b', 'r', 'b']
    comparisons['number_nouns_verbs']['ls'] = ['-', '-', '--', '--']
    comparisons['number_nouns_verbs']['sort'] = ['word_string']
    
    # Pronouns and verbs
    comparisons['number_pronouns_verbs'] = {}
    comparisons['number_pronouns_verbs']['queries'] = ["sentence_length<3 and Declarative==1 and word_position==1 and (word_string in ['he', 'she'])", 
                                                       "sentence_length<3 and Declarative==1 and word_position==1 and (word_string in ['they'])", 
                                                       "sentence_length<3 and Declarative==1 and word_position==2 and pos in ['VBZ']", 
                                                       "sentence_length<3 and Declarative==1 and word_position==2 and pos in ['VBP']"]
    comparisons['number_pronouns_verbs']['condition_names'] = ['Noun Singular', 'Noun Plural', 'Verb Singular', 'Verb Plural', 'Pronoun Singular', 'Pronoun Plural', 'Verb-pro Singular', 'Verb-pro Plural']
    comparisons['number_pronouns_verbs']['colors'] = ['r', 'b', 'r', 'b']
    comparisons['number_pronouns_verbs']['ls'] = ['-', '-', '--', '--']
    comparisons['number_pronouns_verbs']['sort'] = ['num_letters', 'word_string']
   

    # Nouns:
    comparisons['number'] = {}
    comparisons['number']['queries'] = ["grammatical_number==-1 and (pos_simple in ['NN'])", "grammatical_number==1 and (pos_simple in ['NN'])"]
    comparisons['number']['condition_names'] = ['Singular', 'Plural']
    comparisons['number']['colors'] = ['b', 'g']
    comparisons['number']['sort'] = ['word_string']
    comparisons['number']['level'] = 'word'
    comparisons['number']['tmin_tmax'] = [-0.2, 0.6]

    comparisons['number_pronoun'] = {}
    comparisons['number_pronoun']['queries'] = ["word_string in ['he', 'she']", "word_string in ['We', 'they']"]
    comparisons['number_pronoun']['condition_names'] = ['Singular', 'Plural']
    comparisons['number_pronoun']['colors'] = ['b', 'g']
    comparisons['number_pronoun']['sort'] = ['word_string']
    comparisons['number_pronoun']['level'] = 'word'
    comparisons['number_pronoun']['tmin_tmax'] = [-0.2, 0.6]


    # Nouns:
    comparisons['grammatical_number_nouns'] = {}
    comparisons['grammatical_number_nouns']['queries'] = ["sentence_length<4 and word_position==2 and (word_string in ['boy', 'girl', 'man', 'woman', 'host', 'grandpa'])", "sentence_length<4 and word_position==2 and (word_string in ['boys', 'girls', 'men', 'women', 'actors'])"]
    comparisons['grammatical_number_nouns']['condition_names'] = ['Singular', 'Plural']
    comparisons['grammatical_number_nouns']['colors'] = ['b', 'g']

    # Verbs:
    #comparisons['grammatical_number_verbs'] = {}
    #comparisons['grammatical_number_verbs']['queries'] = []
    #comparisons['grammatical_number_verbs']['condition_names'] = ['Singular', 'Plural']
    #comparisons['grammatical_number_verbs']['colors'] = ['b', 'g']

    # Pronouns:
    comparisons['grammatical_number_pronouns'] = {}
    comparisons['grammatical_number_pronouns']['queries'] = ["sentence_length<3 and word_position==1 and (word_string in ['he', 'she'])", "sentence_length<3 and word_position==1 and (word_string in ['they'])"]
    comparisons['grammatical_number_pronouns']['condition_names'] = ['Singular', 'Plural']
    comparisons['grammatical_number_pronouns']['colors'] = ['b', 'g']

    # Pronouns (end of sentence):
    comparisons['grammatical_number_pronouns_end'] = {}
    comparisons['grammatical_number_pronouns_end']['queries'] = ["sentence_length<3 and (sentence_string.str.startswith('he') or sentence_string.str.startswith('she'))", "sentence_length<3 and (sentence_string.str.startswith('they'))"]
    comparisons['grammatical_number_pronouns_end']['condition_names'] = ['Singular', 'Plural']
    comparisons['grammatical_number_pronouns_end']['colors'] = ['b', 'g']
  
# Nouns vs. verbs
    comparisons['nouns_verbs'] = {}
    comparisons['nouns_verbs']['queries'] = ["word_position==2 and (pos=='NN' or pos=='NNS')", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG')"]
    comparisons['nouns_verbs']['condition_names'] = ['Noun', 'Verb']
    comparisons['nouns_verbs']['colors'] = ['b', 'g']
    comparisons['nouns_verbs']['sort'] = ['word_string']

# Gender
    
    comparisons['gender_pronouns'] = {}
    comparisons['gender_pronouns']['queries'] = ["word_position==1 and word_string=='he'", "word_position==1 and word_string=='she'"]
    comparisons['gender_pronouns']['condition_names'] = ['Masculine', 'Feminine']
    comparisons['gender_pronouns']['colors'] = ['b', 'g']

    comparisons['gender_nouns'] = {}
    comparisons['gender_nouns']['queries'] = ["word_position==2 and (word_string=='boy' or word_string=='boys' or word_string=='man' or word_string=='men')", "word_position==2 and (word_string=='girl' or word_string=='girls' or word_string=='woman' or word_string=='women')"]
    comparisons['gender_nouns']['condition_names'] = ['Masculine', 'Feminine']
    comparisons['gender_nouns']['colors'] = ['b', 'g']


# Sentence type    
    comparisons['declarative_questions'] = {}
    comparisons['declarative_questions']['queries'] = ["dec_quest==0", "dec_quest==1"] # and word_position==-1"]
    comparisons['declarative_questions']['condition_names'] = ['Declarative', 'Question']
    comparisons['declarative_questions']['colors'] = ['b', 'g']
    
    
# Embedding
    comparisons['embedding'] = {}
    #comparisons['embedding']['queries'] = ["Embedding==1 and word_position==-1", "Declarative==1 and sentence_length==5 and Embedding==0 and word_position==-1"]
    comparisons['embedding']['queries'] = ["embedding==1", "embedding==0"]
    comparisons['embedding']['condition_names'] = ['Embedded', 'Main']
    comparisons['embedding']['colors'] = ['b', 'g']




# Short vs. Long Words
    comparisons['short_vs_long_words'] = {}
    comparisons['short_vs_long_words']['queries'] = ["word_length<4 and word_length>1", "word_length>4"]
    comparisons['short_vs_long_words']['condition_names'] = ['Short_word', 'Long_word']
    comparisons['short_vs_long_words']['colors'] = ['b', 'g']
    comparisons['short_vs_long_words']['sort'] = ['word_string']

# Word string
    #comparisons['word_string'] = {}
    #comparisons['word_string']['queries'] = "word_string"
    #comparisons['word_string']['condition_names'] = []
    #comparisons['word_string']['colors'] = []
    #comparisons['word_string']['tmin_tmax'] = [-1.2, 1.7]
    #comparisons['word_string']['y-tick-step'] = 20
    
# Word string two -word sentences:
    comparisons['word_string_len2'] = {}
    comparisons['word_string_len2']['queries'] = ["sentence_length<3 and word_position==1 and word_string == 'he'", "sentence_length<3 and word_position==1 and word_string == 'she'", "sentence_length<3 and word_position==1 and word_string == 'they'", "sentence_length<3 and word_position==1 and word_string == 'who'" ]
    comparisons['word_string_len2']['condition_names'] = ['He', 'She', 'They', 'Who']
    comparisons['word_string_len2']['colors'] = ['r', 'g', 'b', 'm']
    comparisons['word_string_len2']['tmin_tmax'] = [-1.2, 1.7]
    comparisons['word_string_len2']['sort'] = ['sentence_string']
    comparisons['word_string_len2']['y-tick-step'] = 3

# Word Starts with
    letters = ['e', 'b', 'h', 'k', 's', 'v', 'j', 'f', 'p', 'm', 'd', 'a', 't', 'i', 'r', 'o', 'u', 'g', 'c', 'l', 'w']
    comparisons[f'word_startswith'] = {}
    comparisons[f'word_startswith']['queries'] = [f"word_string.str.startswith('{letter}')" for letter in letters]
    comparisons[f'word_startswith']['condition_names'] = letters
    comparisons[f'word_startswith']['colors'] = ['b'] * len(letters)

# Word Ends with
    letters = ['e', 'b', 'h', 'k', 's', 'v', 'j', 'f', 'p', 'm', 'd', 'a', 't', 'i', 'r', 'o', 'u', 'g', 'c', 'l', 'w']
    comparisons[f'word_endswith'] = {}
    comparisons[f'word_endswith']['queries'] = [f"word_string.str.endswith('{letter}')" for letter in letters]
    comparisons[f'word_endswith']['condition_names'] = letters
    comparisons[f'word_endswith']['colors'] = ['b'] * len(letters)

# Contains a letter
    letters = ['e', 'b', 'h', 'k', 's', 'v', 'j', 'f', 'p', 'm', 'd', 'a', 't', 'i', 'r', 'o', 'u', 'g', 'c', 'l', 'w']
    comparisons[f'word_contains'] = {}
    comparisons[f'word_contains']['queries'] = [f"word_string.str.contains('{letter}')" for letter in letters]
    comparisons[f'word_contains']['condition_names'] = letters
    comparisons[f'word_contains']['colors'] = ['b'] * len(letters)

# EOS effect
    comparisons['eos'] = {}
    comparisons['eos']['queries'] = ["last_word==True and (block in [1, 3, 5])", "last_word==True and (block in [2, 4, 6])"]
    comparisons['eos']['condition_names'] = ['Visual', 'Auditory'] 
    comparisons['eos']['colors'] = ['b', 'r'] 
    comparisons['eos']['sort'] = ['sentence_length', 'num_letters']
    comparisons['eos']['tmin_tmax'] = [-0.3, 1]
    comparisons['eos']['y-tick-step'] = 15




# VERB TYPE
    comparisons['verb_type'] = {}
    comparisons['verb_type']['queries'] = ["word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Transitive==1", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Unergative==1", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Unaccusative==1", "word_position==2 and (pos=='VB' or pos=='VBZ' or pos=='VBP' or pos=='VBN' or pos=='VBD' or pos=='VBG') and Reflexive==1", "word_position==2 and (pos=='NN' or pos=='NNS')"]
    comparisons['verb_type']['condition_names'] = ['Transitive', 'Unergative', 'Unaccusative', 'Reflexive', 'Noun']
    comparisons['verb_type']['colors'] = ['b', 'c', 'r', 'm', 'k']
    comparisons['verb_type']['sort'] = ['word_string']
    
   
# Phonemes
    # PLACE
    comparisons['place_of_articulation'] = {}
    comparisons['place_of_articulation']['queries'] = []
    comparisons['place_of_articulation']['condition_names'] = []
    for cl in ['BILAB', 'LABDENT', 'DENT', 'ALV', 'PAL', 'LAR']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['"' + s.strip(' ') + '"' for s in curr_class]
        curr_query = "phone_string in [%s]" % (', '.join(curr_class))
        comparisons['place_of_articulation']['queries'].append(curr_query)
        comparisons['place_of_articulation']['condition_names'].append(cl)
    comparisons['place_of_articulation']['colors'] = ['r', 'g', 'b', 'y', 'c', 'm']
    comparisons['place_of_articulation']['sort'] = ['phone_string', 'word_position', 'phone_position']
    comparisons['place_of_articulation']['tmin_tmax'] = [-0.1, 0.4]
    comparisons['place_of_articulation']['y-tick-step'] = 60


    # SONORITY
    comparisons['sonority'] = {}
    comparisons['sonority']['queries'] = []
    comparisons['sonority']['condition_names'] = []
    for cl in ['VLS', 'VOI', 'SON']:
        curr_class = consonants[cl].split(' ')
        curr_class = ['"' + s.strip(' ') + '"' for s in curr_class]
        curr_query = "phone_string in [%s]" % (', '.join(curr_class))
        comparisons['sonority']['queries'].append(curr_query)
        comparisons['sonority']['condition_names'].append(cl)
    comparisons['sonority']['colors'] = ['r', 'g', 'b']
    
    # CONSONANTS VS VOWELS
    comparisons['consonants_vowels'] = {}
    comparisons['consonants_vowels']['queries'] = []
    comparisons['consonants_vowels']['condition_names'] = ['Consonants', 'Vowels']
    all_consonants = ['"'+ph.strip()+'"' for cl in consonants.keys() for ph in consonants[cl].split(' ')]    
    curr_query = "phone_string in [%s]" % (', '.join(all_consonants))
    comparisons['consonants_vowels']['queries'].append(curr_query)
    all_vowels = ['"'+ph.strip()+'"' for cl in vowels.keys() for ph in vowels[cl].split(' ')]    
    curr_query = "phone_string in [%s]" % (', '.join(all_vowels))
    comparisons['consonants_vowels']['queries'].append(curr_query)
    comparisons['consonants_vowels']['colors'] = ['r', 'g']
    
    # PHONE STRING
    comparisons['phone'] = {}
    comparisons['phone']['queries'] = "phone_string"
    comparisons['phone']['condition_names'] = []
    comparisons['phone']['colors'] = [] 
    return comparisons


############################################################################################################
############################################################################################################
############################################################################################################

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
    #CONSONANTS['COR'] = "TH DH T D S Z N CH JH SH ZH Y L R"
    #CONSONANTS['LAB'] = "P B M F V W"
    #CONSONANTS['DOR'] = "K G NG W"
    #CONSONANTS['BILAB'] = "P B M"
    #CONSONANTS['LABDENT'] = "F V"
    #CONSONANTS['DENT'] = "TH DH"
    #CONSONANTS['ALV'] = "T D S Z N"
    #CONSONANTS['PLV'] = "CH JH SH ZH"
    #CONSONANTS['PAL'] = "Y"
    #CONSONANTS['VEL'] = "K G NG"
    #CONSONANTS['LAR'] = "HH"
    # ADDED BY ME
    CONSONANTS['BILAB'] = "P B M"
    CONSONANTS['LABDENT'] = "F V"
    CONSONANTS['DENT'] = "TH DH"
    CONSONANTS['ALV'] = "T D S Z N CH JH SH ZH"
    CONSONANTS['PAL'] = "Y K G NG"
    CONSONANTS['LAR'] = "HH"

    ###CONSONANT MANNER
    CONSONANTS['STOP'] = "P T K B D G"
    #CONSONANTS['AFFR'] = "CH JH"
    CONSONANTS['FRIC'] = "F TH HH V DH"
    CONSONANTS['NAS'] = "M N NG"
    CONSONANTS['LIQ'] = "L R"
    CONSONANTS['SIB'] = "CH S SH JH Z ZH"
    CONSONANTS['GLI'] = "W Y"

    ###CONSONANT VOICING/SONORANCE
    CONSONANTS['VLS'] = "P T K CH F TH S SH HH"
    CONSONANTS['VOI'] = "B D G JH V DH Z ZH"
    CONSONANTS['SON'] = "M N NG L W R Y"

    return VOWELS, CONSONANTS
