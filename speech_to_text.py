
import os
import ssl
import sys
import click
import argparse
import base64
import requests
import nltk
import string
import whisper
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Levenshtein import distance as lev
import nltk.data
import nltk
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction


smoothie = SmoothingFunction().method4

nato_tune=True

nato_list = ['ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO', 'FOXTROT', 'GOLF', 'HOTEL', 'INDIA', 'JULIET', 'KILO',
             'LIMA',
             'MIKE', 'NOVEMBER', 'OSCAR', 'PAPA', 'QUEBEC', 'ROMEO', 'SIERRA', 'TANGO', 'UNIFORM', 'VICTOR',
             'WHISKEY',
             'X-RAY', 'YANKEE', 'ZULU', "Recording license plate", "Reporting license plate",
             "recording license plate", "reporting license plate"]

nato_dict = {'alpha': 'a', 'bravo': 'b', 'charlie':'c', 'delta': 'd', 'echo':'e', 'foxtrot':'f', 'golf':'g',
 'hotel':'h', 'india':'i', 'juliet':'j', 'kilo':'k', 'lima':'l', 'mike':'m', 'november':'m', 'oscar':'o',
'papa': 'p', 'quebec': 'q', 'romeo': 'r', 'sierra':'r', 'tango':'t', 'uniform':'u', 'victor':'v', 'whiskey':'w',
'x-ray':'x', 'yankee':'y', 'zulu':'z'}


class SpeechToTextEngine:
    def __init__(self):
        self.api_base_url = 'https://mcv-testbed.cs.columbia.edu/api'

    def transcribe(self, wav):
        raise Exception('Not implemented yet')

    def get_exp_run(self, id):
        api_url = f'{self.api_base_url}/experiment_run/{id}'
        response = requests.get(api_url)
        if response.status_code == 200:
            resp = response.json()
            return resp
        raise Exception(f'ERROR: Failed to get experiment run {id}')

    def get_exp_run_answer(self, exp_id):
        api_url = f'{self.api_base_url}/experiment/{exp_id}'
        response = requests.get(api_url)
        correct_array = []
        if response.status_code == 200:
            resp = response.json()
            for item in resp["steps"]:
                correct_array.append(str(item["correct_answer"]).lower())
            return correct_array
        raise Exception(f'ERROR: Failed to get experiment {exp_id}'); 


class GoogleSTT(SpeechToTextEngine):
    def __init__(self, opts):
        if opts.get('key', None) is None:
            raise Exception('Missing Google STT key')
        self.key = opts['key']

    def transcribe(self, wav, model=None):
        current_audio = GoogleSTT.get_binary_item_to_based64(self, wav)
        cur_text = GoogleSTT.get_response_by_api_url(self,current_audio)
        start = wav.find("impaired")
        name = wav[start:len(wav)]
        return name , cur_text

    def get_exp_run(self, id):
        return super().get_exp_run(id)

    def get_exp_run_answer(self, exp_id):
        return super().get_exp_run_answer(exp_id)

    def get_binary_item_to_based64(self, audio_cur):
        resp = requests.get(audio_cur)
        return base64.b64encode(resp.content).decode('utf-8')

    def get_response_by_api_url(self, cur_64):
        API_URL = f'https://speech.googleapis.com/v1p1beta1/speech:recognize?key={self.key}'
        global nato_tune
        # print(f"Sending {items} to Google")
        # # doesn't matter if truncated: will know to

        post_request = {
            "config": {
                "encoding": "LINEAR16",
                "languageCode": "en-US",

            },
            "audio": {
                "content": cur_64,
            }
        }
        if nato_tune:
            post_request["config"]["speechContexts"] = [{
                "phrases": [nato_list],
                "boost": 100
            }]
        request = requests.post(API_URL, json=post_request)
        data = request.json()
        try:
            # if request.status_code == 200:
            #     print(data['results'][0]['alternatives'][0]['transcript'])
            #     # print(data)
            plaintext = data['results'][0]['alternatives'][0]['transcript']
            pure_text = str(plaintext).translate(str.maketrans('', '', string.punctuation))
            print("---------------------------------")
            return pure_text
            # else:
            #     raise Exception('Wrong api expid id ')
        except KeyError:
            # raise Exception("Key error ")
            # print(f"Key error caused by {data}")
            # return 0
            print("There is a 15s check for api_ url ")
            return ""


class Whisper(SpeechToTextEngine):
    def transcribe(self, wav, model):
        # whisper.DecodingOptions(fp16=False)
        # backends.mps.is_available()
        # print(torch.backends.mps.is_available())
        device = torch.device("mps")
        device = torch.device("cuda:0")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        w_model = whisper.load_model("base.en")
        if model == 'medium': w_model = whisper.load_model("medium.en")
        elif model == 'large': w_model = whisper.load_model("large")
        result = w_model.transcribe(wav)
        start = wav.find("impaired")
        name = wav[start:len(wav)]
        list_punct = list(string.punctuation)
        pure_text = result["text"].translate(str.maketrans('', '', string.punctuation))
        return name, pure_text

    def get_exp_run(self, id):
        return super().get_exp_run(id)

    def get_exp_run_answer(self, exp_id):
        return super().get_exp_run_answer(exp_id) 


def get_same_items(list1 , list2):
    find_liencse  = []
    for el in list1:
        for cur in list2:
            if el == cur :
                find_liencse.append(cur)
            else:
                continue
    return find_liencse


def get_word_in_dict(ans_dict):
    stpwrd = nltk.corpus.stopwords.words('english')
    smoothie = SmoothingFunction().method4
    new_stopwords = ["reporting", "license", "plate", "putting", "recording", "Reporting", "Supporting", "Recording", "The", "License", "Plate", "life"]
    stpwrd.extend(new_stopwords)
    clear=[]
    pure =[]
    # ans=[]
    for key, item in ans_dict.items():
        text_tokens= word_tokenize(item)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        clear.append((removing_custom_words))
        ans=[]
        for cur in removing_custom_words:
            if cur.isdigit():
                ans.append(cur)
            elif cur.lower() in nato_dict.keys():
                ans.append(cur)
            elif cur.lower() not in nato_dict.keys():
                curcmax= -1
                confidence_dict={}
                for item in nato_dict.keys():
                    confidence_dict[item] = bleu([cur.lower()], item, smoothing_function=smoothie)
                for k, v in confidence_dict.items():
                    needed =  max(list(confidence_dict.values()))
                    if needed == v:
                        ans.append(k)
        pure.append(ans)
    lic=[]
    for item in pure:
        answer_licence = str()
        for cur in item:
            if cur.isdigit():
                answer_licence += (cur)
            else:
                answer_licence += str(cur[0:1].lower())
        lic.append(answer_licence)
    return lic


def get_word_lev(transcribed_answer):
    stpwrd = nltk.corpus.stopwords.words('english')
    smoothie = SmoothingFunction().method4
    new_stopwords = ["reporting", "license", "plate", "putting", "recording", "Reporting", "Supporting", "Recording", "The", "License", "Plate", "life"]
    stpwrd.extend(new_stopwords)
    clear=[]
    pure =[]
    for key, item in transcribed_answer.items():
        text_tokens= word_tokenize(item)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        clear.append((removing_custom_words))
        ans=[]
        for cur in removing_custom_words:
            if cur.isdigit():
                ans.append(cur)
            elif cur.lower() in nato_dict.keys():
                ans.append(cur)
            elif cur.lower() not in nato_dict.keys():
                curcmax= -1
                confidence_dict={}
                for item in nato_dict.keys():
                    score= lev(cur.lower(), item)
                    confidence_dict[item] = 1 - score / max(len(cur.lower()), len(item))
                for k, v in confidence_dict.items():
                    needed =  max(list(confidence_dict.values()))
                    if needed == v:
                        ans.append(k)
        pure.append(ans)
    lic=[]
    for item in pure:
        answer_licence = str()
        for cur in item:
            if cur.isdigit():
                answer_licence += (cur)
            else:
                answer_licence += str(cur[0:1].lower())
        lic.append(answer_licence)
    return lic


def nlp_getliencse(transcribed_answer):
    stpwrd = nltk.corpus.stopwords.words('english')
    new_stopwords = ["reporting", "license", "plate", "putting", "recording", "Reporting", "Supporting", "Recording", "The", "License", "Plate", "life"]
    stpwrd.extend(new_stopwords)
    clear=[]
    for key, item in transcribed_answer.items():
        text_tokens= word_tokenize(item)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        clear.append((removing_custom_words))
    license_list=[]
    license_dirty=[]
    count=0
    for removing_custom_words_cur in clear:
        answer_licence = str()
        answer_licence_dir = str()
        for cur in removing_custom_words_cur:
            if len(removing_custom_words_cur) > 3:
                if cur.isdigit():
                    answer_licence += cur
                    answer_licence_dir += cur
                elif cur.lower() in nato_dict.keys():
                    answer_licence += nato_dict[cur.lower()]
                    answer_licence_dir  += nato_dict[cur.lower()]
                else :
                    answer_licence += "$"+cur+"$"
                    answer_licence_dir += str(cur[0:1].lower())
        count = count +1
        license_list.insert( count, answer_licence)
        license_dirty.insert(count, answer_licence_dir)
    clear_license=[]
    while ("" in license_dirty):
        license_dirty.remove("")
    for i in license_list:
        if "$" not in i and i != '' :
            clear_license.append(i)
    return clear_license , license_dirty


def get_expr_score(current_plate, correct_plate):
    score = 0
    m = len(current_plate)
    lscore = 0
    for i in range (m + 1):
        # if len(current_plate) > 6:
        #     curscore = 5
        #     confidence_score= 1 - curscore / max(len(current_plate), len(correct_plate))
        # else:
        curscore= lev(current_plate, correct_plate)         
        confidence_score= 1 - curscore / max(len(current_plate), len(correct_plate))
        score += confidence_score
        lscore += curscore
    print(f'Experiment Score: {score / m}')
    if lscore / m > 6:
        print ("LScore: 6" )
    else:
        print(f'LScore: {lscore / m}')
    return score / m


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str)
    parser.add_argument('--engine', type=str)
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()
    if args.engine != 'whisper' and args.engine != 'GoogleSTT':
        raise SystemExit('ERROR: The engine must be set to whisper or GoogleSTT.')
    if len(args.id) != 24:
        raise SystemExit('ERROR: The experiment run ID (_id) must be a length of 24 symbols.')
    if args.engine == 'GoogleSTT':
        try:
            api_key = os.environ['GOOGLE_STT_KEY']
            if len(api_key) == 0:
                raise SystemExit('ERROR: Provided Google STT key is not valid')
        except KeyError:
            raise SystemExit('ERROR: Please provide Google STT key in GOOGLE_STT_KEY environment variable')
    if args.engine == 'whisper':
        if args.model != 'base' and args.model != 'medium' and args.model != 'large':
            raise SystemExit('ERROR: Whisper model can be only base, medium or large')

    return args


def get_answers(stt, exp_run_id, model):
    exp_run = stt.get_exp_run(exp_run_id)
    correct_answer = stt.get_exp_run_answer(exp_run['experiment'])
    transcribed_answer = {}
    for wav in exp_run['audio']:
        name, transcription = stt.transcribe(wav=wav, model=model)
        transcribed_answer[name] = transcription
    return transcribed_answer, correct_answer


if __name__ == '__main__':
    args = get_arguments()

    stt = Whisper()
    if args.engine == 'GoogleSTT':
        stt = GoogleSTT({'key': os.environ['GOOGLE_STT_KEY']})

    transcribed_answer, correct_answer = get_answers(stt, args.id, args.model)
    clear_license, license_dirty = nlp_getliencse(transcribed_answer)
    #print("Correct answers")
    #print(correct_answer)

    got_license = get_same_items(list(correct_answer), license_dirty)

    number_of_steps = len(correct_answer)
    print(f'Number of steps: {number_of_steps}')
    correct_rate = len(got_license) / number_of_steps
    print(f'Correct rate: {correct_rate}')

    ans1 = get_word_in_dict(transcribed_answer)
    print(f'Correct rate with Bleu: {len(get_same_items(correct_answer,ans1)) / number_of_steps}')
    #get_expr_score(correct_answer,ans1)
    ans2 = get_word_lev(transcribed_answer)
    print(f'Correct rate with lex: {len(get_same_items(correct_answer, ans2)) / number_of_steps}')

    #get_expr_score(correct_answer, ans2)
    combine = got_license
    combine.extend(ans1)
    print(f'Correct rate with combine: {len(get_same_items(correct_answer, ans2)) / number_of_steps}')
