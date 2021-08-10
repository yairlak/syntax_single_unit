patients = ['502', '504', '505', '510', '513', '515', '530', '538', '539', '540']
#patients = ['502', '504']
data_type = 'micro'
filt = 'raw'


cmd = 'python decode_comparison.py'
for p in patients:
    cmd += f' --patient {p} --data-type {data_type} --filter {filt}'
print(cmd)
