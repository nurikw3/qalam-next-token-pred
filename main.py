from keyboard.keyboard import ChagataiKeyboard
from utils.display import print_suggestions
from model.ngram import NgramModel

kb = ChagataiKeyboard()
# model = NgramModel()
print(kb._parse("Daryo oqimlarini "))
# # должно вернуть (['daryo', 'oqimlarini'], '')
# ctx = ("daryo", "oqimlarini")
# print(ctx in model.trigram)  # есть ли этот биграм как контекст?
# print(model.bigram.get("oqimlarini"))  # что идёт после этого слова?
# print(model.bigram.get("daryo"))  # что идёт после daryo?

# print(("daryo", "oqimlarini") in kb.model.trigram)
# print(kb.model.bigram.get("oqimlarini"))
# print(kb.model.bigram.get("daryo"))

print(kb.model.bigram.get("daryo"))

print(kb.model.trigram.get(("daryo", "sohiliga")))

print(kb.model.trigram.get(("daryo", "sohiliga")))
ctx_norm = ("daryo", "sohiliga")
print(ctx_norm in kb.model.trigram)
while True:
    text = input(">>> ")
    res = kb.suggest_full(text)
    print_suggestions(text, res)
