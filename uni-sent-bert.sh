export PYTHONPATH=$PWD:$PYTHONPATH

python bin/train.py --path-to-corpus-dir='data/dialogue_lined/train' \
                    --path-to-ents-table='data/dialogue_lined/train_entities.csv' \
                    --path-to-saving-model='model/uni-sent-bert' \
                    --bert-name='bert-base-chinese' \
                    --gpu=1 \
                    --mode='uni-sent' \
                    --ideal-batch-size=32 \
                    --actual-batch-size=32 \
                    --max-epochs=12 \
                    --max-input-len=126 \
                    --seed=1