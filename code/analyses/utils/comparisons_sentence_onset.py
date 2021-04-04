def comparison_list():
    comparisons = {}

# Sanity checks:
    comparisons[1] = {}
    comparisons[1]['name'] = 'All_sentences_audio'
    comparisons[1]['train_queries'] = ["word_position==1 and first_phone==1 and (block in [2, 4, 6])"]
    comparisons[1]['train_condition_names'] = ['All_sentences']
    comparisons[1]['colors'] = ['b']

    comparisons[2] = {}
    comparisons[2]['name'] = 'All_sentences_visual'
    comparisons[2]['train_queries'] = ["word_position==1 and (block in [1, 3, 5])"]
    comparisons[2]['train_condition_names'] = ['All_sentences']
    comparisons[2]['colors'] = ['b']


# Sentence type
    
    comparisons[3] = {}
    comparisons[3]['name'] = 'declarative_questions_auditory'
    comparisons[3]['train_queries'] = ["Declarative==1 and (block in [2, 4, 6]) and first_phone==1", "Question==1 and (block in [2, 4, 6]) and first_phone==1"]
    comparisons[3]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[3]['colors'] = ['b', 'g']

    comparisons[4] = {}
    comparisons[4]['name'] = 'declarative_questions_visual'
    comparisons[4]['train_queries'] = ["Declarative==1 and block in [1, 3, 5] and last_word==True", "Question==1 and block in [1, 3, 5] and last_word==True"]
    comparisons[4]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[4]['colors'] = ['b', 'g']

    comparisons[5] = {}
    comparisons[5]['name'] = 'declarative_questions_auditory2visual'
    comparisons[5]['train_queries'] = ["Declarative==1 and block in [2, 4, 6] and last_word==True and first_phone==1", "Question==1 and block in [2, 4, 6] and last_word==True and first_phone==1"]
    comparisons[5]['test_queries'] = ["Declarative==1 and block in [1, 3, 5] and last_word==True", "Question==1 and block in [1, 3, 5] and last_word==True"]
    comparisons[5]['train_condition_names'] = ['Declarative', 'Question']
    comparisons[5]['test_condition_names'] = ['Declarative', 'Question']
    comparisons[5]['colors'] = ['b', 'g']
    
# Embedding

    comparisons[6] = {}
    comparisons[6]['name'] = 'embedding_auditory'
    comparisons[6]['train_queries'] = ["Embedding==1 and last_word==True and first_phone==1 and block in [2, 4, 6]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and first_phone==1 and block in [2, 4, 6]"]
    comparisons[6]['train_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[6]['colors'] = ['b', 'g']

    comparisons[7] = {}
    comparisons[7]['name'] = 'embedding_visual'
    comparisons[7]['train_queries'] = ["Embedding==1 and last_word==True and block in [1, 3, 5]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and block in [1, 3, 5]"]
    comparisons[7]['train_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[7]['colors'] = ['b', 'g']

    comparisons[8] = {}
    comparisons[8]['name'] = 'embedding_audio2visual'
    comparisons[8]['train_queries'] = ["Embedding==1 and last_word==True and first_phone==1 and block in [2, 4, 6]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and first_phone==1 and block in [2, 4, 6]"]
    comparisons[8]['test_queries'] = ["Embedding == 1 and last_word==True and block in [1, 3, 5]", "Declarative==1 and sentence_length==5 and Embedding==0 and last_word==True and block in [1, 3, 5]"]
    comparisons[8]['train_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[8]['test_condition_names'] = ['Embedding', 'Long_declarative']
    comparisons[8]['colors'] = ['b', 'g']
   
    return comparisons


