from keyboard.keyboard import ChagataiKeyboard
from utils.display import print_suggestions

kb = ChagataiKeyboard()

while True:
    text = input(">>> ")
    res = kb.suggest_full(text)
    print_suggestions(text, res)
