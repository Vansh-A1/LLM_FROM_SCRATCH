import tiktoken

tokenizer=tiktoken.get_encoding("gpt2")

text=input("write the encode text you want to encode : ")

ids=tokenizer.encode(text)

print("tokenised sentence :", ids)

decoded_sentence=tokenizer.decode(ids)
print("decoded sentence",decoded_sentence)
