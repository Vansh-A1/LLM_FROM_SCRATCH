from pathlib import Path
import re 

file =Path(__file__).resolve().parents[1] / "archive" / "the-verdict.txt"
with open(file) as files:
    raw_text=files.read()

tokens = re.split(r'[.,:;!?()_\-\s]+', raw_text)
tokens = [t.strip() for t in tokens if t.strip() != '']
tokens=sorted(set(tokens))
vocab={token:idx for idx,token in enumerate(tokens)}

class Tokenizer():
    def __init__(self,vocab):
        self.str_int=vocab
        self.int_str={i:s for s,i in vocab.items()}

    def encoder(self,text):
        tokens = re.split(r'[.,:;!?()_\-\s]+', text)
        tokens = [t.strip() for t in tokens if t.strip() != '']
        ids =[self.str_int[token] for token in tokens]
        return ids
    def decoder(self,ids):
        sentence=" ".join(self.int_str[idx] for idx in ids )
        return sentence

hi=Tokenizer(vocab)
ans=hi.encoder("The height of his glory")
print(ans)
sentence=hi.decoder(ans)
print(sentence)
