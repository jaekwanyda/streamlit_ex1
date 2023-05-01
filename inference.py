import argparse
import os
import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import pytorch_lightning as pl


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name,batch_size, test_file):
        super().__init__()
        self.model_name = model_name
        self.test_file = test_file
        self.batch_size= batch_size

        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<PERSON>']})

    def tokenizing(self, dataframe):
        data = []
        for _, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='predict'):
        # 예측 데이터 준비
        predict_data = self.test_file
        predict_inputs, predict_targets = self.preprocessing(predict_data)
        self.predict_dataset = Dataset(predict_inputs, [])


    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name='snunlp/KR-ELECTRA-discriminator', lr=1e-5, vocab_size=30001):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        self.plm.resize_token_embeddings(vocab_size)

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()


def csv_inference(test_file):

    model_name='snunlp/KR-ELECTRA-discriminator'
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(model_name, batch_size=32, test_file=test_file)
    trainer = pl.Trainer(accelerator='cpu', max_epochs=1, log_every_n_steps=1)
    checkpoint_file='./model/snunlp-KR-ELECTRA-discriminator-sts-val_pearson=0.950-k=0.ckpt'
    model = Model.load_from_checkpoint(checkpoint_file)
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = test_file[['sentence_1','sentence_2','label']]
    output['label'] = predictions
    output=output.to_csv()
    return output
