export PYTHONPATH=$PWD:$PYTHONPATH

python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/dev' \
                                 --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-baseline-further-pretrain-linearLR-lr5e-5-epochs120' \
                                 --path-to-output='multi-sents-baseline-further-pretrain-linearLR-lr5e-5-epochs120.tsv' \
                                 --gpu=0 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --max-input-len=510