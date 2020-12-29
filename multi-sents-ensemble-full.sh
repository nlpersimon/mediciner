export PYTHONPATH=$PWD:$PYTHONPATH


python bin/train.py --path-to-train-corpus-dir='data/dialogue_lined/train' \
                    --path-to-train-ents-table='data/dialogue_lined/train_entities.csv' \
                    --path-to-saving-model='model/multi-sents-test-further-pretrain-lr2.25e-5' \
                    --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                    --bert-name='model/multi-sents-test-further-pretrained-bert' \
                    --gpu=1 \
                    --mode='multi-sents' \
                    --ideal-batch-size=32 \
                    --actual-batch-size=8 \
                    --learning-rate=0.0000225 \
                    --max-epochs=105 \
                    --max-input-len=510 \
                    --seed=1

python bin/train_ensemble.py --path-to-train-corpus-dir='data/dialogue_lined/train' \
                             --path-to-train-ents-table='data/dialogue_lined/train_entities.csv' \
                             --path-to-saving-model='model/multi-sents-ensemble-test-member-lr2.25e-5-self-lr8e-3-ep15' \
                             --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                             --path-to-bert1='model/multi-sents-test-further-pretrain-lr2.25e-5' \
                             --path-to-bert2='model/multi-sents-roberta-wwm-ext-lr2.25e-5' \
                             --gpu=0 \
                             --mode='multi-sents' \
                             --ideal-batch-size=32 \
                             --actual-batch-size=8 \
                             --learning-rate=0.008 \
                             --max-epochs=15 \
                             --max-input-len=510 \
                             --seed=1
