from flask import Flask, render_template, request
from wtforms import (Form, TextField, validators, SubmitField)
from wtforms.validators import Length

import numpy as np
from model_components import Transformer
from tokenizer_helpers import load_tokenizer
from translate_helper_functions import translate, random_translate_from_tensor, plot_attention_weights

# load BPE tokenizers with pre-trained weights
de_tokenizer, de_word_index = load_tokenizer('./tokenizers/tokenizer_de_corpus.json')
en_tokenizer, en_word_index = load_tokenizer('./tokenizers/tokenizer_en_corpus.json')

# define variables "num_tokens_source" and "num_tokens_target"
# these are part of the arguments when creating Transformer object
vocab_len_source = len(de_word_index.keys())
vocab_len_target = len(en_word_index.keys())
num_tokens_source = vocab_len_source + 1
num_tokens_target = vocab_len_target + 1


# load sample data for source and target -- these np arrays will be used 
# to generate random translations
source_train_sample = np.loadtxt('./sample_tensors/source_train_sample.csv', delimiter = ',', dtype = 'int32')
target_train_sample = np.loadtxt('./sample_tensors/target_train_sample.csv', delimiter = ',', dtype = 'int32')

# hard code the "max_source_length" and "max_target_length" using values found during training
max_source_length = 124
max_target_length = 84

# define the arguments that will be fed when creating Transformer object
# these need to be the same as the ones used while training
num_layers = 4
embedding_dim = 64
num_heads = 5
fully_connected_dim = 128
input_vocab_size = num_tokens_source
target_vocab_size = num_tokens_target
max_positional_encoding_input = max_source_length
max_positional_encoding_target = max_target_length

# create Transformer object
transformer = Transformer(
    num_layers=num_layers,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    fully_connected_dim=fully_connected_dim,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_positional_encoding_input = max_positional_encoding_input,
    max_positional_encoding_target = max_positional_encoding_target
    )

# load pre-trained weights of model
file_path = 'saved_models/model'
transformer.load_weights(file_path)


class ReusableForm(Form):
    """User entry form for entering German sentence"""
    
    # input sentence
    input_sent = TextField("Enter a German sentence (max length 100 chars) or write 'random':", 
                            validators=[
                                        validators.InputRequired(),
                                        Length(min = 1, max =100, message = 'Input too long, maximum 100 chars')
                                        ])
    
    # Submit button
    submit = SubmitField("Enter")

app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    
    # Create form
    form = ReusableForm(request.form)

    # After entry on form, and if all conditions are met
    if request.method == 'POST' and form.validate():
        # Extract information
        input_sentence = request.form['input_sent']
        
        # If user chooses to generate translation of random sentence from source data
        if input_sentence == 'random':
            # call "random_translate_from_tensor()" function to generate random input and predicted translation
            pred_sentence, original_sentence, true_translation, attention_weights = random_translate_from_tensor(transformer, 
                                                                                de_tokenizer ,en_tokenizer, 
                                                                                source_train_sample, target_train_sample, 
                                                                                max_target_length)
            
            
            # "plot_attention_weights()" plots the attention scores and saves the plots in the "static" folder
            plot_attention_weights(pred_sentence, original_sentence, attention_weights, de_tokenizer, en_tokenizer)

            # pass info to "random_output.html"
            return render_template('random_output.html', original_sentence = original_sentence, 
                                                        pred_sentence = pred_sentence, 
                                                        true_translation = true_translation)
        
        else:
            # call "translate()" to retrieve model's translation of "input_sentence"
            pred_sentence, source_sentence , attention_weights  = translate(input_sentence, transformer, max_target_length, de_tokenizer, en_tokenizer)
            
            # plot attention scores and save to the "static" folder
            plot_attention_weights(pred_sentence, source_sentence, attention_weights, de_tokenizer, en_tokenizer)
            
            return render_template('pred_output.html', input_sentence = input_sentence, 
                                                        pred_sentence = pred_sentence)


        # Send template information to index.html
    return render_template('index.html', form=form)



if __name__ == '__main__':
    app.run(debug=True)