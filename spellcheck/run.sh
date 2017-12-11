#!/usr/bin/env bash
#
# 
# ------------------------------------------------------------------------------------------------------------------------------------
mkdir ./screen_display_output || true &&
mkdir ./log || true &&
mkdir ./figures || true &&
touch ./screen_display_output/screen_out_0.txt &&
touch ./screen_display_output/screen_out_1.txt &&
touch ./screen_display_output/screen_out_2.txt &&
touch ./screen_display_output/screen_out_4.txt &&
touch ./screen_display_output/screen_out_qwerty.txt &&
current_date_time="`date +%Y-%m-%d,%H:%M:%S`"
echo 'Start running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_0.txt && 
echo 'Start running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_1.txt &&
echo 'Start running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_2.txt &&
echo 'Start running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_4.txt &&
echo 'Start running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_qwerty.txt &&
# 
# ==================================================================
(./experiment-cmd.py 0  2>&1) |  tee -a ./screen_display_output/screen_out_0.txt &
(./experiment-cmd.py 1  2>&1) |  tee -a ./screen_display_output/screen_out_1.txt &
(./experiment-cmd.py 2  2>&1) |  tee -a ./screen_display_output/screen_out_2.txt &
(./experiment-cmd.py 4  2>&1) |  tee -a ./screen_display_output/screen_out_4.txt &
(./qwerty-experiment.py 2>&1) |  tee -a ./screen_display_output/screen_out_qwerty.txt  &
#---------------------
wait
# ----------------- 
current_date_time="`date +%Y-%m-%d,%H:%M:%S`"
echo 'End   running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_0.txt && 
echo 'End   running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_1.txt &&
echo 'End   running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_2.txt &&
echo 'End   running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_4.txt &&
echo 'End   running time(YYYY-MM-DD, HH:MM:SS):'$current_date_time |  tee -a ./screen_display_output/screen_out_qwerty.txt 
#
#
# ----------------------------------------
#
# end of the file 
