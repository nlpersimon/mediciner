export PYTHONPATH=$PWD:$PYTHONPATH

python bin/train.py --path-to-train-corpus-dir='data/dialogue_lined/train' \
                    --path-to-train-ents-table='data/dialogue_lined/train_entities.csv' \
                    --path-to-saving-model='model/multi-sents-test-further-pretrain-crf' \
                    --path-to-vocab='vocab/bert-base-chinese-vocab.txt' \
                    --bert-name='model/multi-sents-test-further-pretrained-bert' \
                    --gpu=0 \
                    --mode='multi-sents' \
                    --ideal-batch-size=32 \
                    --actual-batch-size=8 \
                    --learning-rate=0.00003 \
                    --max-epochs=105 \
                    --max-input-len=510 \
                    --crf \
                    --seed=1
