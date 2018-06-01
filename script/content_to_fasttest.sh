cat ../input/train.csv | cut -f11 -d',' | sed '1d' > context_train.txt
cat ../input/test.csv | cut -f11 -d',' | sed '1d' > context_test.txt

./fastText/fasttext print-sentence-vectors ../input/ru.300.bin < context_train.txt >fasttest_train.csv
./fastText/fasttext print-sentence-vectors ../input/ru.300.bin < context_test.txt > fasttext_test.csv
