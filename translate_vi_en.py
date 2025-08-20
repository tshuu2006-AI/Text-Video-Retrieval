from googletrans import Translator

def translate_vi_to_en(text):
	translator = Translator()
	result = translator.translate(text, src='vi', dest='en')
	return result.text

def translate_vi_to_en(texts):
    translator = Translator()
    if isinstance(texts, list): 
        return [translator.translate(t, src='vi', dest='en').text for t in texts]
    else:  
        return translator.translate(texts, src='vi', dest='en').text


if __name__ == "__main__":
    vi_text = input()
    en_text = translate_vi_to_en(vi_text)
    print(en_text)