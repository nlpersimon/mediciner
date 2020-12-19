export PYTHONPATH=$PWD:$PYTHONPATH

python bin/train.py --path-to-train-corpus-dir='data/dialogue_lined/train' \
                    --path-to-train-ents-table='data/dialogue_lined/train_entities.csv' \
                    --path-to-saving-model='model/multi-sents-roberta-wwm-ext-lr2.25e-5' \
                    --path-to-vocab='vocab/chinese-roberta-wwm-ext-vocab.txt' \
                    --bert-name='hfl/chinese-roberta-wwm-ext' \
                    --gpu=0 \
                    --mode='multi-sents' \
                    --ideal-batch-size=32 \
                    --actual-batch-size=8 \
                    --max-epochs=105 \
                    --learning-rate=0.0000225 \
                    --max-input-len=510 \
                    --seed=1
