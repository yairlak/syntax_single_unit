import sys, os, argparse
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('..')
from utils import utils
import HTML

parser = argparse.ArgumentParser()
parser.add_argument('--patient', default='479_11', nargs='*', help='Patient number')
parser.add_argument('--data-type', default='macro', help='macro/micro/spike')
parser.add_argument('--filter', default='raw', help='raw/high-gamma/gaussian-kernel')
parser.add_argument('--level', default='sentence_onset', choices=['sentence_onset', 'sentence_offset', 'word', 'phone'], help='sentence_onset/sentence_offset/word/phone level')
parser.add_argument('--comparison-name', default='all_trials', help='See functions/comparisons.py')
parser.add_argument('--path2output', default='../../../HTMLs/overview_plots/', help='Destination for html file.')
args = parser.parse_args()
print(args)

##################
# HTML PER PROBE #
##################
per_probe_htmls = []
for patient in args.patient:
    data_type_for_probe_names = 'micro' if args.data_type == 'spike' else args.data_type
    probe_names, _ = utils.get_probe_names(patient, data_type_for_probe_names)
    print(patient, args.data_type, probe_names)
    # GET TEXT
    fn_htmls = []
    for probe_name in probe_names:
        text_list = HTML.HTML_per_probe(patient, args.comparison_name, args.data_type, args.filter, args.level, probe_name)

        # WRITE TO HTML FILE
        fn_html = f'overview_patient_{patient}_{args.comparison_name}_{args.data_type}_{args.filter}_{args.level}_{probe_name}.html'
        fn_html = os.path.join(args.path2output, fn_html)
        with open(fn_html, 'w') as f:
            for line in text_list:
                f.write("%s\n" % line)
        print('HTML saved to: ', fn_html)
        fn_htmls.append(os.path.basename(fn_html)) 
    per_probe_htmls.append([patient, probe_names, fn_htmls])

#########################
# HTML FOR ALL PATIENTS #
#########################
text_list = HTML.HTML_all_patients(per_probe_htmls, args.comparison_name, args.data_type, args.filter, args.level)
# WRITE TO HTML FILE
fn_html = f'All_patients_{args.level}_{args.filter}_{args.data_type}_{args.comparison_name}.html'
fn_html = os.path.join(args.path2output, fn_html)
with open(fn_html, 'w') as f:
    for line in text_list:
        f.write("%s\n" % line)
print('HTML saved to: ', fn_html)

