
python3 file_to_vector_train_desc.py && python3 file_to_vector_test_desc.py && python3 file_to_vector_test.py && python3 file_to_vector_train.py

./fastText/fasttext print-sentence-vectors ../input/ru.300.bin < ../input/train_filter_desc.csv > fasttext_train.csv 

./fastText/fasttext print-sentence-vectors ../input/ru.300.bin < ../input/test_filter_desc.csv > fasttext_test.csv
