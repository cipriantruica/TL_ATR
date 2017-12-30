#!/bin/bash

# run topic modeling using C-value
# change path parameters

PATHR=../results/
DB1=ConferenceDB
DB2=NewsArticlesDB
DB3=ArxivDB
NUM_ITER=200
NUM_TESTS=1
python3.6 main.py $DB1 $NUM_ITER $NUM_TESTS >> $PATHR$DB1"_results"
# python3.6 main.py $DB2 $NUM_ITER $NUM_TESTS >> $PATHR$DB2"_results"
# python3.6 main.py $DB3 $NUM_ITER $NUM_TESTS >> $PATHR$DB3"_results"

