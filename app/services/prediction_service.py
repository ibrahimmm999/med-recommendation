import pandas as pd
from app.models.model import tokenizer, model
from app.utils.preprocessing import preprocess_texts, encode_text, get_similar_text

df = pd.read_excel('training_data.xlsx')

def standardize_output(text):
    filter_1 = text.replace('(', '')
    filter_2 = filter_1.replace(')', '')
    filter_3 = filter_2.replace('-', '')
    filter_4 = filter_3.replace('   ', '')
    return filter_4.replace('  ', ' ')

df['Summary'] = df['Summary'].apply(standardize_output)
df['Drug'] = df['Drug'].apply(standardize_output)

def predict_drug(user_input: str):
    input_ids, attention_mask = encode_text(tokenizer, user_input, 128)
    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
    max_index = np.argmax(predictions)
    
    if max_index == 0:
        category = 'Respiratory diseases'
    elif max_index == 1:
        category = 'Skin diseases'
    else:
        category = 'Gastrointestinal diseases'

    df_search = df[df['Category'] == category]
    input_texts = df_search['Summary'].tolist()
    train_input_ids, train_attention_mask = preprocess_texts(input_texts, tokenizer)
    
    real_word = get_similar_text(user_input, tokenizer, train_input_ids)
    matching_rows = df_search[df_search['Summary'].str.lower().str.contains(real_word, na=False)]['Drug'].tolist()
    
    return category, matching_rows
