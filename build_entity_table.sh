export PYTHONPATH=$PWD:$PYTHONPATH


if [ ! -d "lastday_predictions" ]; then
    mkdir -p "lastday_predictions"
fi

python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/test' \
                                 --path-to-vocab='vocab/chinese-roberta-wwm-ext-large-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-roberta-wwm-ext-large-lr1e-5-ep120' \
                                 --path-to-output='lastday_predictions/test-multi-sents-roberta-wwm-ext-large-lr1e-5-ep120.tsv' \
                                 --gpu=1 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --max-input-len=510
                                 
python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/test' \
                                 --path-to-vocab='vocab/chinese-roberta-wwm-ext-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-roberta-wwm-ext-lr2.25e-5' \
                                 --path-to-output='lastday_predictions/test-multi-sents-roberta-wwm-ext-lr2.25e-5.tsv' \
                                 --gpu=1 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --max-input-len=510
                                 
python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/test' \
                                 --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-test-further-pretrain-lr1e-5' \
                                 --path-to-output='lastday_predictions/test-multi-sents-test-further-pretrain-lr1e-5.tsv' \
                                 --gpu=1 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --max-input-len=510

python bin/build_entity_table.py --path-to-corpus-dir='data/dialogue_lined/test' \
                                 --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                                 --path-to-model-dir='model/multi-sents-ensemble-test-member-lr2.25e-5-self-lr8e-3-ep15' \
                                 --path-to-output='lastday_predictions/test-multi-sents-ensemble-test-member-lr2.25e-5-self-lr8e-3-ep15.tsv' \
                                 --gpu=1 \
                                 --mode='multi-sents' \
                                 --batch-size=8 \
                                 --ensemble \
                                 --max-input-len=510