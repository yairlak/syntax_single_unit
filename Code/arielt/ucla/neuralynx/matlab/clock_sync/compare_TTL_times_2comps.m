%function [] = compare_TTL_times_2comps()
% compare_TTL_times_2comps    

% Author: Ariel Tankus.
% Created: 19.06.2013.


cd /media/arielt/nt1000/arielt/depth/test_256ch/test256ch_computer1;
TTL_per_bit_comp1 = load_nev_multi;

cd /media/arielt/nt1000/arielt/depth/test_256ch/test256ch_computer2/20130619-072429;
TTL_per_bit_comp2 = load_nev_multi;

d_microsec = TTL_per_bit_comp2{1} -  TTL_per_bit_comp1{1};
rng2_microsec = range2(d_microsec);
