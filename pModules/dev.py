
from googletrans import Translator

from konlpy.tag import Mecab
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import gensim.downloader as api

import numpy as np 

import os
import pickle as pk
from collections import defaultdict

class SemanticVerbDict :
    # 한국민족문화대백과사전 참고
    semanticVerbs = {
        '이동동사' : ['가다', '오다', '떠나다'],
        '심리동사' : ['좋다', '나쁘다', '행복하다', '슬프다', '징그럽다', '공허하다', '허무하다'],
        '화행동사' : ['말하다', '명령하다', '제안하다'],
        '단언동사' : ['주장하다', '믿다', '생각하다'],
        '대칭동사' : ['같다', '다르다', '싸우다', '화해하다', '닮다', '만나다'],
        '수헤동사' : ['주다', '드리다', '받다', '얻다', '잃다', '맡기다'],
        '경험동사' : ['알다', '느끼다', '깨닫다', '발견하다', '죽다'],
        '지각동사' : ['보다', '듣다', '맡다'],
        '인지동사' : ['알다', '모르다'],
        '기원동사' : ['원하다', '바라다', '빌다'],
        '재귀동사' : ['입다', '벗다', '먹다', '동작'], 
        '상태동사' : ['시작하다', '멈추다', '그만두다'],
        '수행동사' : ['동작', '행동'] # 위 분류까지 포함되지 않는 일반동사 분류 의도
    }
    def __init__(self, mode='saved') :
        self.path = 'data/'
        self.semanticVerbs = None
        self.semanticSynsets = None

        if not os.path.isdir('data') :
            os.mkdir('data')

        if mode == 'saved' :
            with open(self.path+'engSemanticVerbs.dict', 'rb') as f :
                self.engSemanticVerbs = pk.load(f)

    def get_engSemanticVerbDict(self, return_=True, auto_save=True) :
        import googletrans

        # 구글번역기 API 호출 및 번역
        translator = googletrans.Translator() 
        engSemanticVerbs = dict()
        for key, values in SemanticVerbDict.semanticVerbs.items() :
            lst = list(map(lambda t : translator.translate(t, dest='en').text, values))
            engSemanticVerbs[key] = lst     

        # (사전 탐색한 내용) 번역 내용 수정
        engSemanticVerbs['심리동사'][0] = 'good' # 좋다 : 'good night' -> 'good'
        engSemanticVerbs['지각동사'][-1] = 'smell' # (냄새를)맡다 : 'take on' -> 'smell'
        self.engSemanticVerbs = engSemanticVerbs

        if auto_save :
            with open(self.path+'engSemanticVerbs.dict', 'wb') as f :
                pk.dump(self.engSemanticVerbs, f)
        if return_ : 
            return engSemanticVerbs

    def get_semanticSynsetDict(self, engSemanticVerbs=None, return_=True) :
        if engSemanticVerbs is None :
            engSemanticVerbs = self.engSemanticVerbs

        # 첫번째 유의어로 유의어 단어집 구축
        semanticSynsets = defaultdict(list)
        for cls, verbs in engSemanticVerbs.items() :
            tags = pos_tag(verbs)
            for verb, pos in tags :

                # 구(Phrase) 로 번역된 단어 처리
                if len(verb.split(' ')) > 1 :
                    synsets = wordnet.synsets('_'.join(verb.split(' ')))
                    if len(synsets) == 0 :
                        tokens = word_tokenize(verb)
                        posTokens = pos_tag(tokens)
                        tmp, verb = verb, None
                        for posToken in posTokens :
                            if 'VB' in posToken[1] :
                                verb = posToken[0]
                            if verb is None and 'NN' in posToken[1] :
                                verb = posToken[0]
                        else :
                            if verb is None :
                                print(f"[Error] {tmp} is not valid")
                                raise ValueError
                    else :
                        verb = '_'.join(verb.split(' '))

                try :
                    synset = wordnet.synset(f'{verb}.v.1')
                    semanticSynsets[cls].append(synset)
                except :
                    try :
                        synset = wordnet.synset(f'{verb}.a.1')
                        semanticSynsets[cls].append(synset)
                    except :
                        synset = wordnet.synset(f'{verb}.n.1')
                        semanticSynsets[cls].append(synset)
        self.semanticSynsets = semanticSynsets

        if return_ : 
            return semanticSynsets

def get_engValidVerbs(validVerbs) :
    translator = Translator()
    engValidVerbs = list()
    for p_idx, para_validVerbs in enumerate(validVerbs) :
        engValidVerbs.append([])
        for verb in para_validVerbs :
            transed = translator.translate(verb).text
            # 구(phrase)로 번역된 단어 처리
            if len(transed.split(' ')) > 1 :
                tmpCheck = ''
                for token, pos in pos_tag(transed.split(' ')) :
                    if 'VB' in pos and token not in ['be', 'is', 'was', 'are', 'were'] :
                        tmpCheck = token
                        continue
                    # 우선순위 처리를 위해 조건문 분할
                    if tmpCheck == '' and pos == 'NN' :
                        tmpCheck = token
                        continue
                    if tmpCheck == '' and pos == 'IN' :
                        tmpCheck = token
                        continue
                    if tmpCheck == '' and pos == 'EX' :
                        tmpCheck = token
                        continue                     
                else :
                    if tmpCheck == '' :
                        tmpCheck = 'do' # 일반화
                    engValidVerb = tmpCheck
            else :
                engValidVerb = transed

            engValidVerbs[p_idx].append((verb, engValidVerb))       
    return engValidVerbs

# 'fasttext-wiki-news-subwords-300',
# 'conceptnet-numberbatch-17-06-300',
# 'word2vec-ruscorpora-300',
# 'word2vec-google-news-300',
# 'glove-wiki-gigaword-50',
# 'glove-wiki-gigaword-100',
# 'glove-wiki-gigaword-200',
# 'glove-wiki-gigaword-300',
# 'glove-twitter-25',
# 'glove-twitter-50',
# 'glove-twitter-100',
# 'glove-twitter-200',
# '__testing_word2vec-matrix-synopsis'
def semanticNormalization(engValidVerbs, semanticSynsets, pretrained='glove-twitter-100') :
    
    ### word2vec 테스트 ###
    Word2Vec = api.load(pretrained)
    #####################

    semanticSynsets_list = [
            (key, value)
            for key, values in semanticSynsets.items()
            for value in values
        ]
    lemmatizer = WordNetLemmatizer()
    semanticVerbSeq, semanticClsSeq = list(), list()
    for p_idx, paragraph in enumerate(engValidVerbs) :
        semanticVerbSeq.append([])
        for ko, verb in paragraph :
            verb = lemmatizer.lemmatize(verb, 'v')
            try :
                synset = wordnet.synset(f'{verb}.v.1')
                semanticVerbSeq[p_idx].append((synset, verb))
            except :        
                try :
                    synset = wordnet.synset(f'{verb}.a.1')
                    semanticVerbSeq[p_idx].append((synset, verb))
                except :
                    try :
                        synset = wordnet.synset(f'{verb}.n.1')
                        semanticVerbSeq[p_idx].append((synset, verb))
                    except :
                        synset = wordnet.synsets(verb)[0]
                        semanticVerbSeq[p_idx].append((synset, verb))

            # 유사도 기반 분류
            similarities, tmpSimilarities = list(), list()
            for semanticSynset in semanticSynsets_list :
                similarity = synset.wup_similarity(semanticSynset[1])

                ### word2vec 테스트 ###
                t = semanticSynset[1].name().split('.')[0]
                if len(t.split('_')) > 1 :
                    t = t.split('_')[0]
                similarity = Word2Vec.similarity(verb.lower(), t.lower())
                ######################

                similarities.append(similarity)
                tmpSimilarities.append((similarity, semanticSynset[1]))
            cls = semanticSynsets_list[np.argmax(similarities)][0]
            max_ = semanticSynsets_list[np.argmax(similarities)][1]
            semanticClsSeq.append((cls, verb, ko, max_, similarities[np.argmax(similarities)]))
            
    return semanticClsSeq
        
    