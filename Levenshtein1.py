import random

### JOSE: declare global variables here
exper_id = ""
answer_id = ""
lev_dict = {}
# list_of_attempt_steps
dictAttempts = {}
# list_of_answer_steps
dictAnswers = {}
counter = 0
list1 = []
seen = {}
a = 'TEST123'
b = 'BEST31'


def convertABCD(n, reset):
    global seen, counter
    if reset:
        seen = {}
        counter = 0
    letterlist1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']
    output = ''
    list1 = []
    for letters in n:
        key = letters
        val = letterlist1[counter]
        if letters not in seen:
            counter += 1
            seen[key] = val
            output += val
        else:
            output += seen[letters]
    return output


def lev(a1, b1, convert):
    if len(b1) > len(a1):
        return lev(b1, a1, convert)
    if convert == True:
        a = convertABCD(a1, True)
        b = convertABCD(b1, False)
    else:
        a = a1
        b = b1
    key = a + ", " + b
    if key in lev_dict:
        return lev_dict[key]
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    levtail = lev(a[1:], b[1:], convert)
    if a[0] == b[0]:
        ## put lev_dict[key] = levenshtein distance
        lev_dict[key] = levtail
        return levtail
    levtaila = lev(a[1:], b, convert)
    levtailb = lev(a, b[1:], convert)
    ## put lev_dict[key] = levenshtein distance
    NumVal = lev_dict[key] = 1 + min(levtail, levtaila, levtailb)
    return 1 + min(levtail, levtaila, levtailb)

