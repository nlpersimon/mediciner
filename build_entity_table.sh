export PYTHONPATH=$PWD:$PYTHONPATH

python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/test' \
                                 --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-test-further-pretrain-crf-ep50' \
                                 --path-to-output='test-multi-sents-test-further-pretrain-crf-ep50.tsv' \
                                 --gpu=0 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --crf \
                                 --max-input-len=510