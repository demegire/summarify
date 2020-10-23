import transformers
import torch
from torch import nn
from fastapi import FastAPI
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "saved_model"
abs_file_path = os.path.join(script_dir, rel_path)

app = FastAPI()

tokenizer = transformers.BertTokenizer.from_pretrained(rel_path)

configdict = {
  "_name_or_path": "dbmdz/bert-base-turkish-cased",
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 32000
}

bert_configuration = transformers.PretrainedConfig.from_dict(configdict)

bert_model = transformers.BertModel(bert_configuration)

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = bert_model
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

modelx = SentimentClassifier(6)
modelx.load_state_dict(torch.load("state.pt",  map_location=torch.device('cpu')))
modelx.eval()

@app.get("/items/{query}")
def read_item(query: str):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    review_text = query
    class_names = ['hesap','iade','iptal','kredi','kredi-karti','musteri-hizmetleri']

    encoded_review = tokenizer.encode_plus(
      review_text,
      max_length=200,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      truncation = True,
      return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = modelx(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    return {"item_id": query, "q": class_names[prediction]}