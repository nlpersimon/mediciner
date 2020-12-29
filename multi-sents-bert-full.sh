export PYTHONPATH=$PWD:$PYTHONPATH

python bin/train.py --path-to-train-corpus-dir='data/dialogue_lined/train' \
                    --path-to-train-ents-table='data/dialogue_lined/train_entities.csv' \
                    --path-to-saving-model='model/multi-sents-test-further-pretrain-lr1e-5' \
                    --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                    --bert-name='model/multi-sents-test-further-pretrained-bert' \
                    --gpu=1 \
                    --mode='multi-sents' \
                    --ideal-batch-size=32 \
                    --actual-batch-size=8 \
                    --learning-rate=0.00001 \
                    --max-epochs=105 \
                    --max-input-len=510 \
                    --seed=1
