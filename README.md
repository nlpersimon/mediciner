<h1 align="center">
<p> MediciNER
</h1>

# Installation
建議在 virtual environment 下執行安裝。  
首先參考 pytorch 的[官方網站](https://pytorch.org/)安裝 pytorch，建議安裝 gpu 版本。  
接著在 command line 執行以下指令
```
git clone https://github.com/nccuSimonLee/mediciner.git
pip install -r requirements.txt
```

# Scripts
目前在 bin/ 底下有兩個 python script:
 - train.py: 訓練模型並儲存。
 - train_ensemble.py: 訓練 ensemble 模型並儲存
 - build_entity_table.py: 用訓練好的模型產生給定的文本的 entity table，主要是用來產生測試資料的 entity table，以便上傳至評分系統。

 使用範例請參考 multi-sents-bert-full.sh 以及 build_entity_table.sh。

 # Replicate Final Results
 執行以下的指令復現我們在比賽中的最後上傳答案
 ```
 sh generate_data.sh
 sh further_pretrain.sh
 sh multi-sents-bert-full.sh
 sh multi-sents-roberta-full.sh
 sh multi-sents-ensemble-full.sh
 sh multi-sents-roberta-large-full.sh
 sh build_entity_table.sh
 python aggregate_results.py
 ```

# Main Modules
整個系統由以下三個主要的 module 構成：
 - `dataset`: 負責資料的前處理，將原始資料轉換為輸入模型需要的資料型態。
 - `train`: 負責訓練模型時的細節部份，包含了計算 loss，如何做 validation，以及設定 optimizer。
 - `extractor`: 負責用訓練好的模型，從一篇輸入的原始文本中抽出 entites。

 ## dataset
 `dataset` 這個 module 由三個 submodule 互相協作:
  - `corpus_labeler`: 將一篇原始文本搭配它的 entity 標註，轉為輸入模型需要的 labels。
  - `processor`: 將一篇原始文本拆成多個 `Example`，根據需求還能進一步轉成 `Feature`
    - `Example`: 代表了原始文本中的一句對話或多句對話，它包括了內容、這段內容在原始文本中的 span、這段內容對應的 labels
    - `Feature`: 代表了 `Example` 對應的用來輸入模型的 features，包括了內容對應的 token ids、attention mask、labels
  - `bert_dataset`: 將多篇文本轉為 `Feature`，並 padding 為 tensor。同時也代表了 pytorch 的 custom `Dataset` 實作。

## train
`train` 用來實作 pytorch lightning 的 `LightningModule`:
  - 定義如何計算一個 step (or batch) 的 loss
  - 定義如何做 validation
  - 定義 optimizer 的 setting

## extractor
`extractor` 主要是在利用訓練好的模型產生一段輸入的 labels，並且**對這些 labels 做適當的調整**，比如將 subword label 根據情況調整為 inside label 或 outside label，最後再從中抓出 entities。  
抓出來的 entities 包含的資訊有：
 - start: entity 的第一個字在輸入中的位置
 - end: entity 的最後一個字在輸入中的位置 **+ 1**
 - text: entity 的內容
 - type: entity 的類型