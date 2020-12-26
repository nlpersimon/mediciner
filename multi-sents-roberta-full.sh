export PYTHONPATH=$PWD:$PYTHONPATH

python bin/train.py --path-to-train-corpus-dir='data/dialogue_lined/train' \
                    --path-to-train-ents-table='data/dialogue_lined/train_entities.csv' \
                    --path-to-saving-model='model/multi-sents-roberta-wwm-ext-large-lr1e-5-ep120' \
                    --path-to-vocab='vocab/chinese-roberta-wwm-ext-large-vocab.txt' \
                    --bert-name='hfl/chinese-roberta-wwm-ext-large' \
                    --gpu=1 \
                    --mode='multi-sents' \
                    --ideal-batch-size=32 \
                    --actual-batch-size=2 \
                    --max-epochs=120 \
                    --learning-rate=0.00001 \
                    --max-input-len=510 \
                    --seed=1
