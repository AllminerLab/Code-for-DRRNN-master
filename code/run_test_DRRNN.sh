python ./run.py \
-r ../data/Instrument/Musical_Instruments_5.json \
-d ../data/Instrument/input_data/ \
-rt ../data/Instrument/preprocessed/rating \
-re ../data/Instrument/preprocessed/review \
-o ../data/Instrument/result/DRRNN \
-m 300 \
-p ../data/glove/glove.6B.300d.txt \
-f 128 \
-c True


python ./run.py \
-r ../data/beauty/Beauty_5.json \
-d ../data/beauty/input_data/ \
-rt ../data/beauty/preprocessed/rating \
-re ../data/beauty/preprocessed/review \
-o ../data/beauty/result/DRRNN \
-m 300 \
-p ../data/glove/glove.6B.300d.txt \
-f 16 \
-c True

python ./run.py \
-r ../data/Automotive/Automotive_5.json \
-d ../data/Automotive/input_data/ \
-rt ../data/Automotive/preprocessed/rating \
-re ../data/Automotive/preprocessed/review \
-o ../data/Automotive/result/DRRNN \
-p ../data/glove/glove.6B.300d.txt \
-m 300 \
-f 16 \
-c True


python ./run.py \
-r ../data/Patio/Patio_Lawn_and_Garden_5.json \
-d ../data/Patio/input_data/ \
-rt ../data/Patio/preprocessed/rating \
-re ../data/Patio/preprocessed/review \
-o ../data/Patio/result/DRRNN \
-m 300 \
-p ../data/glove/glove.6B.300d.txt \
-f 32 \
-c True
