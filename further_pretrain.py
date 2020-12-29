from tokenizers import BertWordPieceTokenizer
from transformers import (BertForMaskedLM,
                          LineByLineTextDataset,
                          BertTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments,
                          pipeline)



def main():
    tokenizer = BertTokenizer.from_pretrained('vocab/bert-base-chinese-vocab.txt')

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="data/dialogue_lined/multi-sents-further-pretrain/train_test_dialogues.txt",
        block_size=512,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir="model/multi-sents-test-further-pretrained-bert",
        do_train=True,
        warmup_steps=int(100 * (len(dataset) / 32) * 0.1),
        #warmup_steps=10000,
        overwrite_output_dir=True,
        num_train_epochs=100,
        #max_steps=100000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        save_steps=1000,
        logging_steps=10,
        weight_decay=0.01
    )

    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )
    
    trainer.train()
    
    trainer.save_model('model/multi-sents-test-further-pretrained-bert')
    
    return

if __name__ == '__main__':
    main()