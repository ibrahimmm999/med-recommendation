from keras.models import load_model
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Layer
import tensorflow as tf

class MedBERTEmbeddingLayer(Layer):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', **kwargs):
        super(MedBERTEmbeddingLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = TFAutoModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def get_config(self):
        config = super(MedBERTEmbeddingLayer, self).get_config()
        config.update({
            'model_name': self.model_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_medbert_model(model_path: str):
    with tf.keras.utils.custom_object_scope({'MedBERTEmbeddingLayer': MedBERTEmbeddingLayer}):
        model = load_model(model_path)
    return model

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
model = load_medbert_model('category_classification_model-weight.h5')
