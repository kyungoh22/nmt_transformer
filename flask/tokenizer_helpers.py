import re
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing



# def preprocess_sentence(sentence):
"""
version that removes all punctuation
"""
#     sentence = sentence.lower()
#     sentence = re.sub("'", '', sentence)
#     sentence = sentence.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')
#     exclude = set(string.punctuation)
#     sentence = ''.join(ch for ch in sentence if ch not in exclude)
#     sentence = 'start_ ' + sentence + ' _end'
#     sentence = sentence.encode("ascii", "ignore")
#     sentence = sentence.decode()
#     return sentence



def preprocess_sentence(sentence):
    
    # replace umlauts and sharp s
    sentence = sentence.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')
    sentence = sentence.replace('Ü', 'Ue').replace('Ä', 'ae').replace('Ö', 'Oe')
    # add space before and after punctuation
    sentence = re.sub(r"\b([.,!?()':;])\b", ' \g<0> ', sentence)
    # collapse multiple spaces into single space
    sentence = re.sub('\s{2,}', ' ', sentence)
    sentence = ''.join(ch for ch in sentence)
    # encode ascii and ignore unencodable characters; this removes non-english characters
    sentence = sentence.encode("ascii", "ignore")
    # decode back to utf-8 (which is default of the "encoding" parameter)
    sentence = sentence.decode()
    return sentence


def create_train_tokenizer(file):

    """
    Creates token-index mapping as a json file

    Argument:
    file = list with single element – must end with .txt
    """
    # initialise tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # train on file, adding two tokens
    tokenizer.train(files = file, 
                    special_tokens= ['_____', 'start_', '_end'])        # '_____' added as special token to ensure 'start_' has index 1
                                                                        # will later remove '_____' so that no token has index 0
    
    # get just the file name – select everything that comes after last slash ('/')
    file_name = file[0].rsplit('/', 1)[1]
    
    tokenizer.save(f'tokenizers/tokenizer_{file_name[:-4]}.json')                    # save as json instead of txt
    print(f'json file saved at: tokenizers/tokenizer_{file_name[:-4]}.json')


def load_tokenizer(json_file):
    """
    Loads pre-trained tokenizer from json file

    Argument: json_file path
    Returns: tokenizer, dict_word_index
    """

    tokenizer = Tokenizer.from_file(json_file)                          # load tokenizer

    # add "start_" and "_end" tokens to every tokenized sequence
    tokenizer.post_processor = TemplateProcessing(
    single="start_ $A _end",                                        # $A represents input sentence
    special_tokens=[
        ("start_", tokenizer.token_to_id("start_")), ("_end", tokenizer.token_to_id("_end")),],)
    
    dict_word_index = tokenizer.get_vocab()
    dict_word_index.pop('_____')            # remove '_____' from dictionary – now indexing starts from 1

    return tokenizer, dict_word_index


