export PYTHONPATH=$PWD:$PYTHONPATH



mkdirIfNotExist()
{
    if [ ! -d $1 ]; then
    mkdir -p $1
    fi
}

mkdirIfNotExist "data"
mkdirIfNotExist "data/original/train"
mkdirIfNotExist "data/original/sample"
mkdirIfNotExist "data/original/dev"
mkdirIfNotExist "data/original/test"
mkdirIfNotExist "data/dialogue_lined/train"
mkdirIfNotExist "data/dialogue_lined/sample"
mkdirIfNotExist "data/dialogue_lined/dev"
mkdirIfNotExist "data/dialogue_lined/test"
mkdirIfNotExist "data/dialogue_lined/multi-sents-further-pretrain/"

python raw_to_original.py
python original_to_dialogue-lined.py
python generate_fp_data.py