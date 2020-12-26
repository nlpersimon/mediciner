export PYTHONPATH=$PWD:$PYTHONPATH

python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/test' \
                                 --path-to-vocab='vocab/bert-base-multilingual-cased-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-multiling-lr2.25e-5' \
                                 --path-to-output='test-multi-sents-multiling-lr2.25e-5.tsv' \
                                 --gpu=1 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --max-input-len=510