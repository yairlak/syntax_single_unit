from functions import comparisons

# COMPARISON
comparisons = comparisons.comparison_list()

comparison_names = comparisons.keys()
for comp_name in comparison_names:
    print('-'*100)
    print(comp_name)
    for (cond_name, q) in zip(comparisons[comp_name]['condition_names'], comparisons[comp_name]['queries']):
        print(cond_name, ':',  q)

print('-'*100)
print('ALL COMPARISON NAMES')
print(comparison_names)
print()

#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('--level', default = 'word')
#args = parser.parse_args()
