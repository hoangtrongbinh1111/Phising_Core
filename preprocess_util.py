"""# START REGION: ĐỊNH NGHĨA HÀM"""
import pandas as pd
#import socket
import matplotlib.pyplot as plt
import numpy as np
#import pylab as pl
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
#import Evaluation metrics 
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
import json
import tldextract
import sys
#import domain_parser
import whois
from datetime import date
import requests
import csv
import os
import time
#from bs4 import BeautifulSoup
#from urllib.parse import urlencode

"""# Check xem đã được Index chưa"""

"""# Tách url ra các thành phần nhỏ hơn: domain, subdomain, path"""

def words_raw_extraction(domain,subdomain,path):
    
    w_domain = re.split("\-|\.|\/|\?|\=|\@|&|\%|\:|\_|\+",domain.lower())
    w_subdomain = re.split("\-|\.|\/|\?|\=|\@|&|\%|\:|\_|\+",subdomain.lower())
    w_path = re.split("\-|\.|\/|\?|\=|\@|&|\%|\:|\_|\+",path.lower())
    
    raw_words = w_domain+w_path+w_subdomain
    
    raw_words = list(filter(None,raw_words))
    
    return raw_words

"""# Thuộc tính kiểu nlp"""

import re
import sys
import pickle
import numpy as np
import editdistance
from traceback import format_exc

class nlp_class:

    def __init__(self):
        self.path_data = ""
        self.name_keywords = "./data/keywords.txt"
        self.name_brand_file = "./data/allbrands.txt"
        self.name_random_model = "./data/gib_model.pki"

        model_data = pickle.load(open(self.name_random_model, 'rb'))
        self.model_mat = model_data['mat']
        self.threshold = model_data['thresh']

        self.keywords = self.__txt_to_list(open(self.name_keywords, "r"))

        self.allbrand = self.__txt_to_list(open(self.name_brand_file, "r"))

    def __txt_to_list(self, txt_object):
        list = []

        for line in txt_object:
            list.append(line.strip())
        txt_object.close()
        return list

    def __is_similar_to_any_element(self, word, list):

        target = ''
        for l in list:
            if editdistance.eval(word, l) < 2:
                target = l

        if len(target) > 0:
            return word
        else:
            return 0

    def parse(self, words):

        keywords_in_words = []
        brands_in_words = []
        similar_to_brands = []
        similar_to_keywords = []
        dga_in_words = []
        len_gt_7 = []
        len_lt_7 = []
        try:
            for word in words:

                word = re.sub("\d+", "", word)

                if word in self.keywords:
                    keywords_in_words.append(word)

                elif word in self.allbrand:
                    brands_in_words.append(word)

                elif self.__is_similar_to_any_element(word, self.allbrand) != 0:
                    target = self.__is_similar_to_any_element(word, self.allbrand)
                    similar_to_brands.append(target)

                elif self.__is_similar_to_any_element(word, self.keywords) != 0:
                    target = self.__is_similar_to_any_element(word, self.keywords)
                    similar_to_keywords.append(target)

                elif len(word) > 3 and not word.isnumeric():

                    if (avg_transition_prob(word, self.model_mat) > self.threshold) == False:
                        dga_in_words.append(word)
                        # todo keyword benzeri olanlar temizlenmeli
                    elif len(word) < 7:
                        len_lt_7.append(word)
                    else:
                        len_gt_7.append(word)

            result = {'keywords_in_words': keywords_in_words, 'brands_in_words': brands_in_words,
                      'dga_in_words': dga_in_words, 'len_lt_7': len_lt_7, 'len_gt_7': len_gt_7,
                      'similar_to_brands': similar_to_brands, 'similar_to_keywords': similar_to_keywords}
        except:
            print(str(words) + " işlenirken hata")
            print("Error : {0}".format(format_exc()))

        return result

    def fraud_analysis(self, grouped_words, splitted_words):

        word_list = grouped_words['len_lt_7'] + grouped_words['similar_to_brands'] + grouped_words[
            'similar_to_keywords'] + splitted_words

        word_list_nlp = grouped_words['len_lt_7'] + grouped_words['similar_to_brands'] + \
                        grouped_words['similar_to_keywords'] + grouped_words['brands_in_words'] + \
                        grouped_words['keywords_in_words'] + grouped_words['dga_in_words'] + splitted_words

        found_keywords = []
        found_brands = []
        similar_to_keyword = []
        similar_to_brand = []
        other_words = []
        target_words = {'brand': [], 'keyword': []}
        try:
            for word in word_list:

                word = re.sub("\d+", "", word)

                if word in self.keywords:
                    found_keywords.append(word)
                elif word in self.allbrand:
                    found_brands.append(word)
                else:

                    for brand in self.allbrand:
                        if editdistance.eval(word, brand) < 2:
                            target_words['brand'].append(brand)
                            similar_to_brand.append(word)

                    for keyword in self.keywords:
                        if editdistance.eval(word, keyword) < 2:
                            target_words['keyword'].append(keyword)
                            similar_to_keyword.append(word)

                if word not in found_keywords+found_brands+similar_to_keyword+similar_to_brand:
                    other_words.append(word)

            result = {'found_keywords': found_keywords,
                      'found_brands': found_brands,
                      'similar_to_keywords': similar_to_keyword,
                      'similar_to_brands': similar_to_brand,
                      'other_words': other_words,
                      'target_words': target_words,
                      'words_nlp': word_list_nlp}
        except:
            print(str(word_list)+" işlenirken hata")
            print("Error : {0}".format(format_exc()))
        return result

    def evaluate(self, grouped_words, fraud_analyze_result, splitted_words):

        """
        grouped_words
        keywords_in_words, brands_in_words,
        dga_in_words, len_lt_7, len_gt_7 
        fraud_anaylze_result
        found_keywords, found_brands,
        similar_to_keyword, similar_to_brand,
        other_words, target_words 
        """
        try:
            words_raw = grouped_words['keywords_in_words'] + grouped_words['brands_in_words'] + \
                        grouped_words['similar_to_brands'] + grouped_words['similar_to_keywords'] + \
                        grouped_words['dga_in_words'] + grouped_words['len_lt_7'] + grouped_words['len_gt_7']

            words_len = []
            compound_word_len = []

            for word in words_raw:
                words_len.append(len(word))

            for word in grouped_words['len_gt_7']:
                compound_word_len.append(len(word))

            all_keywords = grouped_words['keywords_in_words'] + fraud_analyze_result['found_keywords']
            all_brands = grouped_words['brands_in_words'] + fraud_analyze_result['found_brands']
            similar_brands = fraud_analyze_result['similar_to_brands']
            similar_keywords = fraud_analyze_result['similar_to_keywords']

            if len(compound_word_len) == 0:
                av_com = 0
            else:
                av_com = float(np.average(compound_word_len))

            if len(words_len) == 0:
                min = 0
                max = 0
                av_w = 0
                std = 0
            else:
                min = int(np.min(words_len))
                max = int(np.max(words_len))
                av_w = float(np.average(words_len))
                std = float(np.std(words_len))

            result = {'info': {'keywords': all_keywords,
                               'brands': all_brands,
                               'dga_in_words': grouped_words['dga_in_words'],
                               'similar_to_keywords': similar_keywords,
                               'similar_to_brands': similar_brands,
                               'negligible_words': fraud_analyze_result['other_words'],
                               'target_words': fraud_analyze_result['target_words'],
                               'words_nlp': fraud_analyze_result['words_nlp']
                               },
                      'features': {'raw_word_count': len(words_len),
                                   'splitted_word_count': len(splitted_words),
                                   'average_word_length': av_w,
                                   'longest_word_length': max,
                                   'shortest_word_length': min,
                                   'std_word_length': std,
                                   'compound_word_count': len(grouped_words['len_gt_7']),
                                   'keyword_count': len(all_keywords),
                                   'brand_name_count': len(all_brands),
                                   'negligible_word_count': len(fraud_analyze_result['other_words']),
                                   'target_brand_count': len(fraud_analyze_result['target_words']['brand']),
                                   'target_keyword_count': len(fraud_analyze_result['target_words']['keyword']),
                                   'similar_keyword_count': len(similar_keywords),
                                   'similar_brand_count': len(similar_brands),
                                   'average_compound_words': av_com,
                                   'random_words': len(grouped_words['dga_in_words'])
                                   }}
        except:
            print("Error : {0}".format(format_exc()))
        return result

    def check_word_random(self, word):

        if avg_transition_prob(word, self.model_mat) < self.threshold:
            return 1
        else:
            return 0

"""# Tách kiểu Word"""

import re
import pprint
import enchant

class WordSplitterClass(object):

    def __init__(self):

        self.path_data = ""
        self.name_brand_file = "./data/allbrands.txt"
        self.dictionary_en = enchant.DictWithPWL("en_US", self.name_brand_file)
        #self.__file_capitalize(self.path_data, self.name_brand_file)

        self.pp = pprint.PrettyPrinter(indent=4)

    def _split(self, gt7_word_list):

        return_word_list = []

        for word in gt7_word_list:
            try:
                ss = {'raw': word,'splitted':[]}

                # kelime içerisinde rakam varsa temizlenir.
                word = re.sub("\d+", "", word)
                sub_words = []

                if not self.dictionary_en.check(word):
                    #  işlenen kelime sözlükte bu kelimeyi geri döndür. İşlenen kelime sözlükte yoksa ayırma işlemine geç.

                    for number in range(len(word), 3, -1): # uzunluğu 3 den yüksek olan alt kelimelerin üretilmesi
                        for l in range(0, len(word) - number + 1):
                            if self.dictionary_en.check(self.__capitalize(word[l:l + number])):

                                #  bir kelime tespit ettiğim zaman diğer kelimelerin tespit edilmesinde fp ye sebep olmasın diye
                                #  tespit edilen kelime yerine * karekteri koydum
                                w = self.__check_last_char(word[l:l + number])
                                sub_words.append(w)
                                word = word.replace(w, "*" * len(w))

                    rest = max(re.split("\*+", word), key=len)
                    if len(rest) > 3:
                        sub_words.append(rest)

                    split_w = sub_words

                    for l in split_w:
                        for w in reversed(split_w):

                            """
                            tespit edilen bir kelime daha büyük olan bir kelimenin içerisinde de geçiyorsa o fp dir.
                            Bunları temizledim. Örn.  secure, cure.
                            Cure kelimesi listeden çıkarılır.
                            """

                            if l != w:  # todo edit distance eklenecek
                                if l.find(w) != -1 or l.find(w.lower()) != -1:
                                    sub_words.remove(w)

                    if len(sub_words) == 0:
                        #  eğer hiç kelime bulunamadıysa ham kelime olduğu gibi geri döndürülür.
                        sub_words.append(word.lower())
                else:
                    sub_words.append(word.lower())

                ss['splitted']=sub_words
                return_word_list.append(ss)
            except:
                print("|"+word+"| işlenirken hata")
                print("word_splitter.split()  muhtemelen boş dizi gelme hatası  /  Error : {0}".format(format_exc()))

        return return_word_list

    def _splitl(self, gt7_word_list):

        result = []

        for val in self._split(gt7_word_list):
            result += val["splitted"]

        return result

    def _splitw(self, word):

        word_l = []
        word_l.append(word)

        result = self._split(word_l)

        return result

    def __check_last_char(self, word):

        confusing_char = ['s', 'y']
        last_char = word[len(word)-1]
        word_except_last_char = word[0:len(word)-1]
        if last_char in confusing_char:
            if self.dictionary_en.check(word_except_last_char):
                return word_except_last_char

        return word

    def __clear_fp(self, sub_words):

        length_check = 0
        for w in sub_words:
            if (length_check + len(w)) < self.length+1:
                length_check = length_check + len(w)
            else:
                sub_words.remove(w)

        sub_words = self.__to_lower(sub_words)
        return sub_words

    def __to_lower(self, sub_words):

        lower_sub_list = []

        for w in sub_words:
            lower_sub_list.append(str(w.lower()))

        return lower_sub_list

    def __capitalize(self, word):
        return word[0].upper() + word[1:]

    def __file_capitalize(self, path, name):

        """
        enchant paketinde özel kelimelerin kontrol edilebilmesi için baş harfinin büyük olması gerekiyor.
        bir kelime kontrol edilmeden önce baş harfi büyük hale gitirilip sonra sözlüğe sorduruyorum.
        Bu nedenle dosyadaki kelimelerin de baş harflerini büyük hale getirip kaydettim ve bu şekilde kullandım.
        :return: 
        """

        personel_dict_txt = open("{0}{1}".format(path, name), "r")

        personel_dict = []

        for word in personel_dict_txt:
            personel_dict.append(self.__capitalize(word.strip()))

        personel_dict_txt.close()

        personel_dict_txt = open("{0}{1}-2".format(path, name), "w")

        for word in personel_dict:
            personel_dict_txt.write(word+"\n")

import math
import pickle

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation,
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    """ Return all n grams from l after normalizing """
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])

def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

"""# Các thuộc tính tự định nghĩa"""

#from domain_parser import domain_parser
import math
import re


def parse_url(url_data):
  try:
    extract_result = tldextract.extract(url_data)
    return (extract_result.subdomain,extract_result.domain,extract_result.suffix)#domain_parser.parse_domain(url_data)
  except Exception as e:
    return ("","","")

def digit_count(data):
  data =str(data)
  return sum(c.isdigit() for c in data)

def get_length(data):
  return len(data)

def issubdomainwww(x):
    if x == 'www':
        return True
    return False

def tld_check(tld):
    if tld is None:
        return False
    common_tld = ["com", "org", "net", "de", "edu", "gov"]
    common_tld = ['com', 'at', 'uk', 'pl', 'be', 'biz', 'br', 
                  'ca', 'cc', 'cn', 'co', 'jp', 'co_jp', 'de',
                  'cz', 'de', 'eu','edu', 'fr', 'id', 'info', 
                  'io', 'it', 'kr', 'ru', 'lv', 'me', 'gov', 
                  'mobi', 'mx', 'name', 'net', 'nyc', 'nz', 
                  'online', 'org', 'pharmacy', 'press', 'pw',
                  'store', 'rest', 'ru_rf', 'security', 'sh', 
                  'site', 'space', 'tech', 'tel', 'theatre', 
                  'tickets', 'tv', 'us', 'uz', 'video', 'website', 'wiki', 'xyz']
    
    common_tld = ['com', 'biz', 'net', 'info', 'ru', 'website', 'online', 'us',
       'vip', 'co', 'me', 'org', 'cl', 'co.uk', 'id', 'vn', 'in',
       'lt', 'co.id', 'by', 'com.br', 'eu', 'pk', 'com.bd', 'ga', 'es',
       'ch', 'com.pl', 'com.my', 'lk', 'pt', 'com.pe', 'no', 'gq',
       'com.au', 'hosting', 'vu', 'ac.in', 'go.id', 'pro', 'de', 'pl',
       'web.id', 'cc', 'com.vn', 'se', 'cn', 'edu', 'mobi', 'com.pk',
       'gr', 'ca', 'ie', 'org.mx', 'tv', 'biz.tr', 'win', 'cf', 'com.sa',
       'com.ua', 'co.za', 'mx', 'waw.pl', 'ro', 'co.nz', 'org.br', 'be',
       'it', 'hu', 'com.tr', 'my', 'desi', 'at', 'com.jo', 'xyz',
       'com.co', 'com.ar', 'fi', 'tools', 'edu.ua', 'edu.bd', 'com.np',
       'sg', 'asia', 'ma', 'nl', 'com.ng', 'io', 'com.tw', 'edu.pe',
       'com.ph', 'org.uk', 'szczecin.pl', 'ge', 'co.in', 'gov.lk', 'nf',
       'fr', 'co.ke', 'cz', 'club', 'ir', 'ac', 'dk', 'tech', 'edu.pl',
       'ua', 'systems', 'com.gr', 'ws', 'org.au', 'co.il', 'ae', 'com.mx',
       'lv', 'tk', 'com.ve', 'md', 'la', 'rs', 'kz', 'com.uy', 'aero',
       'or.id', 'link', 'edu.lb', 'co.ao', 'sk', 'top', 'or.ke', 'in.ua',
       'co.rs', 'rec.br', 'download', 'site', 'com.bo', 'am', 'sch.id',
       'gov.bd', 'net.au', 'mn', 'nu', 'edu.in', 'org.ng', 'com.bt', 'af',
       'si', 'co.th', 'uz', 'bm', 'net.gr', 'lu', 'gob.pe', 'org.za',
       'com.py', 'com.sg', 'co.kr', 'ink', 'RU', 'or.tz', 'su',
       'solutions', 'ly', 'co.zw', 'com.cn', 'sl', 'adv.br', 'org.ph',
       'bz', 'org.pk', 'bg', 'nyc', 'com.fj', 'org.bd', 'net.tr',
       'com.mk', 'zp.ua', 'edu.vn', 'ac.id', 'edu.ng', 'solar', 'sch.sa',
       'cloud', 'edu.lk', 'net.in', 'org.ua', 'net.br', 'uk', 'ml',
       'com.mo', 'hk', 'edu.co', 'ind.br', 'org.in', 'fm', 'name',
       'gov.in', 'ac.zm', 'art.br', 'org.nz', 'com.pt', 'med.br', 'pe',
       'host', 'pw', 'org.py', 'inf.br', 'org.ve', 'zgora.pl', 'today',
       'com.do', 'edu.ec', 'edu.cn', 'ps', 'edu.pk', 'com.kh', 'cat',
       'im', 'ph', 'al', 'org.rs', 'net.pl', 'mw', 'edu.np', 'psc.br',
       'or.kr', 'uy', 'kiev.ua', 'kr', 'edu.rs', 'edu.br', 'gov.ph',
       'COM', 'ec', 'run', 'cab', 'ba', 'radio.br', 'com.ec',
       'wroclaw.pl', 'ind.in', 'co.mz', 'tax', 'com.es', 'services',
       'co.cr', 'com.ge', 'edu.ar', 're', 'xn--p1ai', 'com.cy', 'edu.au',
       'mk', 'cm', 'cx', 'ee', 'kg', 'co.jp', 'com.ro', 'prato.it', 'rw',
       'com.gt', 'mu', 'ms', 'is', 'com.ba', 'moe', 'space', 'world',
       'loan', 'co.at', 'one', 'dn.ua', 'life', 'ns.ca', 'com.hk',
       'sp.gov.br', 'tw', 'cd', 'bf', 'review', 'hr', 'gdn', 'accountant',
       'center', 'com.gh', 'media', 'av.tr', 'digital', 'sy', 'help',
       'co.tz', 'info.pl', 'net.pk', 'org.bw', 'com.lb', 'gob.ve',
       'in.rs', 'jp', 'dp.ua', 'gov.za', 'org.np', 'int', 'fin.ec', 'bid',
       'org.pe', 'az', 'ac.cn', 'edu.jm', 'mg', 'date', 'i.ng', 'gov.pg',
       'mobi.ng', 'desa.id', 'malopolska.pl', 'edu.mx', 'eng.br']

    
    if tld in common_tld:
        return True
    return False

def special_chars(data):
  special_char = {'-': 0, ".": 0, "/": 0, '@': 0, '?': 0, '&': 0, '=': 0, "_": 0}
  special_char_letter = special_char.keys()
  for l in data:
    if l in special_char_letter:
      special_char[l] = special_char[l] + 1
  return (special_char['-'],special_char['.'],special_char['/'],special_char['@'],special_char['?'],special_char['&'],special_char['='],special_char['_'])

def sum_special_chars(data):
  special_char = {'-': 0, ".": 0, "/": 0, '@': 0, '?': 0, '&': 0, '=': 0, "_": 0}
  special_char_letter = special_char.keys()
  for l in data:
    if l in special_char_letter:
      special_char[l] = special_char[l] + 1
  return math.fsum((special_char['-'],special_char['.'],special_char['/'],special_char['@'],special_char['?'],special_char['&'],special_char['='],special_char['_']))

def get_protocal(data):
    if data.find('://') == -1:
        return 'None'
    else:
        domain = data.split("://")[0]
        if domain == 'https':
            return 'https'
        elif domain == 'http':
            return 'http'
    
    return 'None'

def word_process(data):
    line = data['url']
    extracted_domain = data['tld']
    tmp = line[line.find(extracted_domain):len(line)]  # tld sonraki ilk / e gore parse --> path
    pth = tmp.partition("/")
    domain = pth[1] + pth[2]
    return domain
    
from datetime import date

def get_website_details(data):
    
    try:
        line = data['domain'] +'.'+ data['tld']
        
        domain = whois.query(line)
        if domain == None or domain.__dict__ == None:
            raise Exception('None', 'No')
        dicti = domain.__dict__
        
        dict_value ={}
        
        if dicti['creation_date'] is not None:
            dict_value['creation_date'] = dicti['creation_date'].strftime('%Y-%m-%d')
        else:
            dict_value['creation_date'] = None
            
        if dicti['expiration_date'] is not None:
            dict_value['expiration_date'] = dicti['expiration_date'].strftime('%Y-%m-%d')   
        else:
            dict_value['expiration_date'] = None
        
        if dicti['last_updated'] is not None:
            dict_value['last_updated'] = dicti['last_updated'].strftime('%Y-%m-%d') 
        else:
            dict_value['last_updated'] = None

        return (dict_value['creation_date'],dict_value['expiration_date'],dict_value['last_updated'],dicti['registrar'])
    except Exception as e:
        date_value = date.today().strftime('%Y-%m-%d')
        #print(e)
        return (date_value,date_value,date_value,"")

"""# Hàm chính xử lý dataset"""

def substract_dates(date_value):
    
    date_val_creation = (date_value['today_date']-date_value['creation_date']).days
        
    date_val_expiration = abs((date_value['today_date']-date_value['expiration_date']).days)
        
    date_val_last_updated = (date_value['today_date']-date_value['last_updated']).days
    
    if(date_val_creation == 0):
        date_val_creation =  -1
        
    if(date_val_expiration == 0):
        date_val_expiration =  -1
        
    if(date_val_last_updated == 0):
        date_val_last_updated =  -1
        
        
    return (date_val_creation,date_val_expiration,date_val_last_updated)

def count_special_characters_in_domain(data):
    sub_domain = [data['subdomain'],data['domain'],data['tld']]
    
    special_char = {'-': 0, ".": 0, "/": 0, '@': 0, '?': 0, '&': 0, '=': 0, "_": 0}
    
    for data in sub_domain:
        special_char_letter = special_char.keys()
        for l in data:
            if l in special_char_letter:
                special_char[l] = special_char[l] + 1
                
    return (special_char['-'],special_char['.'],special_char['/'],special_char['@'],special_char['?'],special_char['&'],special_char['='],special_char['_'])

def get_NLP_features(data):
    
    try:

        s = data['url'].strip().replace('"',"").replace("'",'')


        extracted_domain = tldextract.extract(s)

        tmp = s[s.find(extracted_domain.suffix):len(s)]

        pth = tmp.partition("/")

        words_raw = words_raw_extraction(extracted_domain.domain,extracted_domain.subdomain,pth[2])


        nlp_manager = nlp_class()

        word_splitter = WordSplitterClass()

        grouped_words = nlp_manager.parse(words_raw)
        splitted_words =  word_splitter._splitl(grouped_words['len_gt_7'])

        fraud_analyze_result = nlp_manager.fraud_analysis(grouped_words, splitted_words)

        result = nlp_manager.evaluate(grouped_words, fraud_analyze_result, splitted_words)
        split = {'raw': grouped_words['len_gt_7'], 'splitted': splitted_words}
        result['info']['compoun_words'] = split

        val = result['features']
        
        return (val['raw_word_count'],val['splitted_word_count'],val['average_word_length'],val['longest_word_length'],val['shortest_word_length'],
           val['std_word_length'],val['compound_word_count'],val['keyword_count'],val['brand_name_count']
           ,val['negligible_word_count'],val['target_brand_count'],val['target_keyword_count'],val['similar_keyword_count'],
           val['similar_brand_count'],val['average_compound_words'],val['random_words'])
    except Exception as e:
        print(e)
        return (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

from sklearn import preprocessing

#INPUT: raw data
#OUTPUT: extracted informations (aka features) from that data
def process_url(data):
    url_dataset = data.copy()
    url_dataset['url_length'] = url_dataset['url'].apply(len)
    url_dataset[['subdomain','domain','tld']] = url_dataset['url'].apply(parse_url).apply(pd.Series)

    url_dataset['tld_digit_count'] = url_dataset['tld'].apply(digit_count)
    url_dataset['domain_digit_count'] = url_dataset['domain'].apply(digit_count)
    url_dataset['subdomain_digit_count'] = url_dataset['subdomain'].apply(digit_count)
    
    url_dataset['tld_length'] = url_dataset['tld'].apply(get_length)
    url_dataset['subdomain_length'] = url_dataset['subdomain'].apply(get_length)
    url_dataset['domain_length'] = url_dataset['domain'].apply(get_length)

    url_dataset['isKnown_tld'] = url_dataset['tld'].apply(tld_check)
    url_dataset['issubdomainwww'] = url_dataset['subdomain'].apply(issubdomainwww)
    url_dataset['count_of_special_char'] = url_dataset['url'].apply(sum_special_chars)
    url_dataset['protocol'] = url_dataset['url'].apply(get_protocal)
    url_dataset[['creation_date','expiration_date','last_updated','registrar']] = url_dataset.apply(get_website_details,axis=1).apply(pd.Series)    
    
    
    url_dataset['word_process'] = url_dataset.apply(word_process,axis=1)
    
    url_dataset['count .'] = url_dataset['url'].str.count('.')
    url_dataset['count -'] = url_dataset['url'].str.count('-')
    url_dataset['count @'] = url_dataset['url'].str.count('@')
    url_dataset['count //'] = url_dataset['url'].str.count('//')
    
    #url_dataset['count -'] = 0
    #url_dataset['count .'] = 0
    #url_dataset['count /'] = 0
    #url_dataset['count @'] = 0
    #url_dataset['count ?'] = 0
    #url_dataset['count &'] = 0
    #url_dataset['count ='] = 0
    #url_dataset['count _'] = 0

    #url_dataset[['count -','count .','count /','count @','count ?','count &','count =','count _']] = url_dataset['url'].apply(special_chars).apply(pd.Series)

    url_dataset['today_date'] = pd.to_datetime(date(2019,11,5))#data collected date
    url_dataset['creation_date'] = pd.to_datetime(url_dataset['creation_date'])
    url_dataset['expiration_date'] = pd.to_datetime(url_dataset['expiration_date'])
    url_dataset['last_updated'] = pd.to_datetime(url_dataset['last_updated'])

    url_dataset[['creation_date_days','expiration_date_days','last_updated_days']] = url_dataset.apply(substract_dates,axis=1).apply(pd.Series)    

    url_dataset['creation_date_days'].fillna(-1,inplace=True)
    url_dataset['expiration_date_days'].fillna(-1,inplace=True)
    url_dataset['last_updated_days'].fillna(-1,inplace=True) 
    
    url_dataset[['domaincount -','domaincount .','domaincount /','domaincount @','domaincount ?','domaincount &','domaincount =','domaincount _']] = url_dataset.apply(count_special_characters_in_domain,axis=1).apply(pd.Series) 

    url_dataset[['raw_word_count', 'splitted_word_count', 'average_word_length', 'longest_word_length', 'shortest_word_length', 'std_word_length', 'compound_word_count', 'keyword_count', 'brand_name_count', 'negligible_word_count', 'target_brand_count', 'target_keyword_count', 'similar_keyword_count', 'similar_brand_count', 'average_compound_words', 'random_words']] =url_dataset.apply(get_NLP_features, axis=1).apply(pd.Series)    

    le = preprocessing.LabelEncoder()
    le.fit(url_dataset['registrar'].apply(str))
    url_dataset['registrar_encoded']=le.transform(url_dataset['registrar'].apply(str))
    return url_dataset
"""# END REGION: ĐỊNH NGHĨA HÀM"""