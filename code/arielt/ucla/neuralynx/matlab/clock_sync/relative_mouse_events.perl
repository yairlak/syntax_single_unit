#!/usr/bin/perl --

# Author: Ariel Tankus.
# Created: 30.08.2009.


$state = 0;     # 0: remember the current init value.  1: o/w.
while (<>) {
    ($sec, $micsec, $remainder) = split(/ /, $_, 3);
    if ($state == 0) {
        if ($sec == int($sec)) {
            $init_val = $sec*1E6 + $micsec;
        } else {
            $init_val = $sec*1E6;  # first field has decimal point XXXXX.yyyy
        }
        $state = 1;
    }
    if ($sec == int($sec)) {
        # was:
#       printf "%d %s", $sec*1E6 + $micsec - $init_val, $remainder;
        printf "%d %s", $sec*1E6 + $micsec, $remainder;
    } else {
        printf "%d %s %s", $sec*1E6, $micsec, $remainder; # actually, $micsec is the first string, not a numeric field.
    }
}
