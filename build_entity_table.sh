export PYTHONPATH=$PWD:$PYTHONPATH

python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/dev' \
                                 --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                                 --path-to-model-dir='model/uni-sent-bert' \
                                 --path-to-output='model11.tsv' \
                                 --gpu=1 \
                                 --mode='uni-sent' \
                                 --batch-size=32 \
                                 --max-input-len=126