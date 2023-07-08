from spacy_extractor.EntityExtraction import *
#from spacy_extractor.EntityExtraction import cleanString

inputString = 'a.i.'

entityString = ''

intermediate_result = inputString.split(' ')

for string in intermediate_result:
    tokenText = string.lstrip(''.join(punctuations2))
    if tokenText.startswith('#') or tokenText.startswith('@'):
        entityString += cleanString(tokenText) + " "
    else:
        entityString += (string.strip(''.join(punctuations2)).lower() if len(re.findall('\.', string))==1 else string) + " "

print(inputString)
print(entityString)



