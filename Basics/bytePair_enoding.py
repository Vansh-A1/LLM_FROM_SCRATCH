import tiktoken

tokenizer=tiktoken("gpt2")

text=input("write the encode text you want to encode")

ids=toeknizer.encode(text)

decoded_sentence=tokenizer.decode(ids)
