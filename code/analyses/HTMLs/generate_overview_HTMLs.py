import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils import utils
import HTML


path2output = '../../../HTMLs/overview_plots/'
comparison_names = ['all_trials', 'all_trials_chrono']
filter_types = ['raw', 'high-gamma']

#######################
# HTML FOR COMPARISON #
#######################
text_list = HTML.HTML_comparison_names(comparison_names)
# WRITE TO HTML FILE
fn_html = f'overview_all_comparisons.html'
fn_html = os.path.join(path2output, fn_html)
os.makedirs(os.path.dirname(fn_html), exist_ok=True)
with open(fn_html, 'w') as f:
    for line in text_list:
        f.write("%s\n" % line)
print('HTML saved to: ', fn_html)

for comparison_name in comparison_names:
    ######################
    # HTML FOR DATA-TYPE #
    ######################
    text_list = HTML.HTML_data_types(comparison_name)
    # WRITE TO HTML FILE
    fn_html = f'All_data_types_{comparison_name}.html'
    fn_html = os.path.join(path2output, fn_html)
    with open(fn_html, 'w') as f:
        for line in text_list:
            f.write("%s\n" % line)
    print('HTML saved to: ', fn_html)
    
    for data_type in ['macro', 'micro', 'spike']:
        ###################
        # HTML FOR FILTER #
        ###################
        text_list = HTML.HTML_filters(comparison_name, data_type)
        # WRITE TO HTML FILE
        fn_html = f'All_filters_{data_type}_{comparison_name}.html'
        fn_html = os.path.join(path2output, fn_html)
        with open(fn_html, 'w') as f:
            for line in text_list:
                f.write("%s\n" % line)
        print('HTML saved to: ', fn_html)

        for filt in filter_types:
            ##################
            # HTML FOR LEVEL #
            ##################
            text_list = HTML.HTML_levels(comparison_name, data_type, filt)
            # WRITE TO HTML FILE
            fn_html = f'All_levels_{filt}_{data_type}_{comparison_name}.html'
            fn_html = os.path.join(path2output, fn_html)
            with open(fn_html, 'w') as f:
                for line in text_list:
                    f.write("%s\n" % line)
            print('HTML saved to: ', fn_html)
