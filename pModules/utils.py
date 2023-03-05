import os
import json
import pickle as pk
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from konlpy.tag import Mecab
from googletrans import Translator

import re
from selenium import webdriver
from selenium.webdriver.common.by import By

class NovelpiaScrapper :
    baseURL = lambda genre_id, page_id=1 : f'https://novelpia.com/plus/all/view/{page_id}/?main_genre={genre_id}'
    novelURL = lambda novel_id : f'https://novelpia.com/novel/{novel_id}'
    epiURL = lambda epi_id : f'https://novelpia.com/viewer/{epi_id}'
    genres = {'romance' : 6}
    features = ['title', 'novel_id', 'epi_num', 'epi_title', 'epi_id', 'text']
    n_split = 3 # 명시적 선언
    # 3 : 상위, 중위, 하위 
    # 추가 세분화 시, self._split_novel_set_by_hit() 수정 필요

    def __init__(self, saved=True, path=None) :
        self.saved = saved
        self.path = 'data/' if path is None else path
        self.mapping_novel_list = None

        if self.saved :
            with open(self.path+'mapping_novel_list.dict', 'rb') as f :
                self.mapping_novel_list = pk.load(f)    

        if not os.path.isdir('data') :
            os.mkdir('data')

    def get_scrappedCSV(self, genre='romance', k=18, n_epi=10, auto_save=True) :
        """ 
        args notice ::
         `k(=18)` : 상위k개, 중위k개, 하위k개 (10% 권장, 간격을 두고 중위데이터들 추출하기 위한 목적)

        """
        SCRAP = self._scrapping(genre, k, n_epi)
        SCRAP['category'] = np.arange(NovelpiaScrapper.n_split).repeat(k*n_epi) 
        scrappedCSV = pd.DataFrame(SCRAP)
        
        if auto_save :
            now_time = datetime.now()
            month, day = now_time.month, now_time.day 
            scrappedCSV.to_csv(self.path+f'm{month}d{day}++{genre}++n{k}epi{n_epi}.csv')
            
        return scrappedCSV

    def _scrapping(self, genre='romance', k=18, n_epi=10) :
        assert genre in NovelpiaScrapper.genres.keys(), f'genre.{genre} is not in {NovelpiaScrapper.genres.keys()}'

        genre_id =  NovelpiaScrapper.genres[genre]
        URL = NovelpiaScrapper.baseURL(genre_id)
        driver = webdriver.Safari()
        driver.get(URL) # 입력 장르의 유료작품 리스트 페이지 이동

        SCRAP = {feature:list() for feature in NovelpiaScrapper.features}
        
        if self.mapping_novel_list is not None :
            mapping_novel_list = self.mapping_novel_list
        else :
            mapping_novel_list = self._get_novel_list_by_genre(driver, genre_id)

        split_ = self._split_novel_set_by_hit(driver, mapping_novel_list, k)
        for novel_ids in split_ :
            tmpSCRAP = self._scrap_episodes(driver, novel_ids, n_epi, mapping_novel_list)
            for feature in NovelpiaScrapper.features :
                SCRAP[feature] += tmpSCRAP[feature]

        return SCRAP

    def _get_novel_list_by_genre(self, driver, genre_id) :
        last_page_link = driver.find_elements(By.CLASS_NAME, 'page-link')[-1].get_attribute('href')
        last_page_id = int(last_page_link.split('/')[-2]) # 마지막 리스트 페이지 아이디
        
        mapping_novel_list = dict()
        for page_id in range(1, last_page_id+1) :
            driver.get(NovelpiaScrapper.baseURL(genre_id, page_id))
            items = driver.find_elements(By.CLASS_NAME, 'cut_line_one')
            for item in items :
                novel_id = item.get_attribute('onclick').split('/')[-1][:-2]
                mapping_novel_list[novel_id] = item.text

        with open(self.path+'mapping_novel_list.dict', 'wb') as f :
            pk.dump(mapping_novel_list, f)

        return mapping_novel_list        
            
    def _split_novel_set_by_hit(self, driver, mapping_novel_list, k) :
        novel_ids = list(mapping_novel_list.keys())
        topK = novel_ids[:k]
        middle = novel_ids[k+k:len(novel_ids)-(k+k)]
        np.random.shuffle(middle)
        middleK = middle[:k]
        bottomK = novel_ids[-1:-(k+1):-1]

        return topK, middleK, bottomK

    def _scrap_episodes(self, driver, novel_ids, n_epi, mapping_novel_list) :
        tmpSCRAP = {feature:list() for feature in NovelpiaScrapper.features}
        
        for n_idx, novel_id in enumerate(novel_ids) :
            title = mapping_novel_list[novel_id]

            novelURL = NovelpiaScrapper.novelURL(novel_id)
            driver.get(novelURL) # 작품 에피소드 리스트 페이지로 이동

            lst = driver.find_elements(By.ID, 'episode_list')[0]
            epis = lst.find_elements(By.CSS_SELECTOR, 'td.font12') # 에피소드 1회 부터 추출

            epi_ids, epi_nums, epi_titles = list(), list(), list()
            for iter, epi in enumerate(epis) :
                if epi.text.split(' ')[0].replace('\t', '').strip() == 'PLUS' : 
                    if iter < 5 : print(f'[WARNING] {title} epi < 5')
                    break
        
                epi_num = epi.text.split('\n\t\t\t\t')[3].strip()
                epi_title = epi.text.split('\n\t\t\t\t')[2].strip()
                epi_id = epi.get_attribute('onclick').split("'")[-2].split('/')[-1]  
                epi_nums.append(epi_num)
                epi_titles.append(epi_title)
                epi_ids.append(epi_id)

            for idx, epi_id in enumerate(epi_ids[:n_epi]) :
                epiURL = NovelpiaScrapper.epiURL(epi_id)
                driver.get(epiURL) # 에피소드 페이지로 이동

                epi_num = epi_nums[idx]
                epi_title = epi_titles[idx]
                text = driver.find_element(By.ID, 'novel_drawing').text # 텍스트 데이터 추출
                
                extracted = [title, novel_id, epi_num, epi_title, epi_id, text]
                for feature, data in zip(NovelpiaScrapper.features, extracted) : 
                    tmpSCRAP[feature].append(data)
                    
        return tmpSCRAP          

class TextPreprocessor : 
    def __init__(self, path=None) :
        self.path = path if path is not None else 'data/'
        if not os.path.isdir('data') :
            os.mkdir('data')

    def preprocessing(self, scrappedCSV, auto_save=True) :
        ppCSV = scrappedCSV.copy()
        ppData = defaultdict(list)
        for idx, episode in ppCSV.iterrows() :
            text = episode.text
            sentences = self.split_sentence(text)
            ppSentences = self.basicPreprocessing(sentences)
            paraINFO, epiINFO = self.get_contentINFO(ppSentences)
            validActions = self.get_actions(ppSentences)
            
            nValidActions = list(map(lambda v : [len(v)], validActions))
            epiINFO['nValidAction'] = int(np.sum(nValidActions))
            epiINFO['validActionRatio'] = epiINFO['nValidAction']/epiINFO['n_sentence']
            epiINFO['vaRatio'] = epiINFO['n_va']/epiINFO['n_sentence']
            epiINFO['vvRatio'] = epiINFO['n_vv']/epiINFO['n_sentence']
            paraINFO['nValidAction'] = nValidActions
            paraINFO['validActionSeq'] = validActions
            
            epiINFO = json.dumps(epiINFO)
            paraINFO = json.dumps(paraINFO)

            ppData['episodeINFO'].append(epiINFO)
            ppData['paragraphINFO'].append(paraINFO)

        for info, data in ppData.items() :
            ppCSV[info] = data
            
        if auto_save :
            ppCSV.to_csv(self.path+'ppCSV.csv')

        return ppCSV

    def split_sentence(self, text) :
        sentences = text.replace('\xa0\xa0\xa0','').replace('\xa0','').split('\n')
        sentences = list(map(lambda sentence : sentence.strip(), sentences))
        return sentences

    def basicPreprocessing(self, sentences) :
        ppSentences, paragraphs, tmpScript = list(), ['<문단시작>'], ''
        silence_check, script_check = 0, 0
        for sentence in sentences :
            if sentence == '' : 
                if silence_check >= 3 and paragraphs[-1] != '<침묵>' : # 연속된 4번째 개행일 경우 침묵 처리(연속된 침묵은 생략)
                    paragraphs.append('<침묵>')
                    silence_check = 0
                else :
                    silence_check += 1
            else :
                silence_check = 0
                if '커버보기' in sentence or '다음화 보기' in sentence : continue
                if '작가의 한마디' in sentence : break 

                # 문단 분리(작품마다 기준이 다름)
                p = re.compile('\d+$|\*+$|\#+$|\-+$') # 문단 구분자 설정
                if p.match(sentence) : # 문단 구분자일 경우
                    if len(paragraphs) : # 문단 데이터가 있을 경우
                        paragraphs.append('<문단끝>')
                        ppSentences.append(paragraphs)
                        paragraphs = ['<문단시작>']

                # 문단 내 문장 처리
                else :
                    # 대화가 한 개행이전에 완전히 끝나지 않은 경우 식별해 하나의 문장으로
                    invalid_script_start = re.compile('^\".+[^\"]$|^\'.+[^\']$|^“.+[^”]$|^‘.+[^’]$')
                    invalid_script_end = re.compile('^[^\'].+\'$|^[^\"].+\"$|^[^“].+”$|^[^‘].+’$')
                    if invalid_script_start.match(sentence) :
                        tmpScript += sentence
                        script_check = 1
                        continue
                    if script_check :
                        tmpScript += ' '+sentence
                        if invalid_script_end.match(sentence) : 
                            paragraphs.append(tmpScript)
                            tmpScript = ''
                            script_check = 0
                        continue

                    paragraphs.append(sentence)

        # 마지막 문단 처리                
        paragraphs.append('<문단끝>')        
        ppSentences.append(paragraphs) 

        return ppSentences  

    def get_contentINFO(self, ppSentences) :
        paraINFO = defaultdict(list)
        epiINFO = {
            'episodeLen' : 0,
            'n_sentence' : 0,            
            'n_script' : 0,
            'n_narration' : 0,
            'n_va' : 0,
            'n_vv' : 0,            
            'sentenceLenSeq' : list(),
            'scriptLenSeq' : list(), 
            'narrationLenSeq' : list(),
        }
        tokenizer = Mecab()
        script_filter = re.compile('^\".+\"$|^\'.+\'$|^“.+”$|^‘.+’$')
        for p_idx, para in enumerate(ppSentences) :
            tmpParagraphLen, tmp_n_sentence, tmp_n_script, tmp_n_narration = 0, 0, 0, 0
            tmpSentenceLenSeq, tmpScriptLenSeq, tmpScripts, tmpNarrationLenSeq, tmpNarrations = list(), list(), list(), list(), list()
            tmp_n_vaSeq, tmp_n_vvSeq = list(), list()
            for sentence in para :
                epiINFO['episodeLen'] += len(sentence)
                epiINFO['n_sentence'] += 1
                epiINFO['sentenceLenSeq'].append(len(sentence))
                tmpParagraphLen += len(sentence)
                tmpSentenceLenSeq.append(len(sentence))
                tmp_n_sentence += 1
                if script_filter.match(sentence) : # 인용구 처리가 된 문장들(대사처리)
                    epiINFO['n_script'] += 1
                    epiINFO['scriptLenSeq'].append(len(sentence))
                    tmp_n_script += 1
                    tmpScriptLenSeq.append(len(sentence))
                else :
                    epiINFO['n_narration'] += 1
                    epiINFO['narrationLenSeq'].append(len(sentence))
                    tmp_n_narration += 1
                    tmpNarrationLenSeq.append(len(sentence))  

                tokens = tokenizer.pos(sentence)
                va = [token for token in tokens if 'VA' in token[1] or 'XSA' in token[1] or 'ETM' in token[1]]
                vv = [token for token in tokens if 'VV' in token[1] or 'XSV' in token[1]]
                tmp_n_vaSeq.append(len(va))
                tmp_n_vvSeq.append(len(vv))
                epiINFO['n_va'] += len(va)
                epiINFO['n_vv'] += len(vv)

            paraINFO['paragraphLen'].append([tmpParagraphLen])
            paraINFO['sentenceLenSeq'].append(tmpSentenceLenSeq)
            paraINFO['n_sentence'].append([tmp_n_sentence])
            paraINFO['n_script'].append([tmp_n_script])
            paraINFO['scriptLenSeq'].append(tmpScriptLenSeq)
            paraINFO['n_narration'].append([tmp_n_narration])
            paraINFO['narrationLenSeq'].append(tmpNarrationLenSeq)
            paraINFO['n_vaSeq'].append(tmp_n_vaSeq)
            paraINFO['n_vvSeq'].append(tmp_n_vvSeq)

            # nan 발생 방지
            mean_sentenceLen = np.sum(tmpSentenceLenSeq)/(len(tmpSentenceLenSeq)+1e-8)
            mean_scriptLen = np.sum(tmpScriptLenSeq)/(len(tmpScriptLenSeq)+1e-8)
            mean_narrationLen = np.sum(tmpNarrationLenSeq)/(len(tmpNarrationLenSeq)+1e-8)
            paraINFO['mean_sentenceLen'].append(np.mean(mean_sentenceLen))
            paraINFO['mean_scriptLen'].append(np.mean(mean_scriptLen))
            paraINFO['mean_narrationLen'].append(np.mean(mean_narrationLen))
        

        for c in ['sentence', 'script', 'narration'] :
            # 문장길이 평균(nan 발생 방지)
            epiINFO[f'mean_{c}Len'] = np.sum(epiINFO[f'{c}LenSeq'])/(len(epiINFO[f'{c}LenSeq'])+1e-8)

            # 문장길이 변화(차분합)
            diff = np.diff(epiINFO[f'{c}LenSeq'])[10:] # 도입부 문장 길이가 대체로 짧아 왜곡 방지를 위해 
            epiINFO[f'{c}LenDiff'] = int(np.sum(diff))

            # 대화, 서술 비율
            if c in ['script', 'narration'] :
                epiINFO[f'{c}_ratio'] = epiINFO[f'n_{c}']/epiINFO['n_sentence']

        epiINFO['nParagrah'] = len(paraINFO['paragraphLen'])

        return paraINFO, epiINFO

    def get_actions(self, ppSentences, onlyNarration=True) :
        tokenizer = Mecab()
        validPosSeqs = ['NNG', 'NNG+XSV', 'NNG+XSV+EP', 'NNG+XSV+EP+EF', 'NNG+XSV+EF', 'VV', 'VV+EP', 'VV+EP+EF', 'VV+EC', 'VV+EC+VX', 'VV+EC+VX+EP', 'VV+EC+VX+EP+EF']
        validActions = list()
        for p_idx, paragraph in enumerate(ppSentences) :
            tmpValidActions = list()
            for sentence in paragraph : 
                tokens = tokenizer.pos(sentence)
                tmpSeq, tmpCheck = '', ''
                tmpStartSeq, tmpStartCheck = '', '' 
                for token in tokens :
                    if token[1] == 'VV+EP' :
                        token = token[0]+'다'
                        if len(token) == 2 : continue
                        tmpValidActions.append(token)
                        continue

                    if token[1] in ['VV', 'NNG'] :
                        tmpSeq += token[0]
                        tmpCheck += token[1]
                        continue
                    
                    if tmpCheck in validPosSeqs :
                        tmpCheck += '+'+token[1]
                        if tmpCheck in validPosSeqs :
                            tmpSeq += token[0]
                            if tmpCheck in ['NNG+XSV+EP+EF', 'NNG+XSV+EF', 'VV+EP+EF', 'VV+EC+VX+EP+EF'] :
                                if tmpSeq != '있었다' :
                                    tmpValidActions.append(tmpSeq)
                                tmpSeq, tmpCheck = '', ''
                            continue
                        else :
                            tmpSeq, tmpCheck = '', ''
                            continue

            validActions.append(tmpValidActions)                
        return validActions        
    

class MeanSeqExtraction :
    def __init__(self) :
        self.cum_ = None
        self.count_ = None

    def __call__(self, dataset:pd.DataFrame, category:int, mode:str) :
        """
        Args ::
            * `category:int` : 작품 분류 인덱스(0:상위, 1:중위, 2:하위)
            * `mode:str` : 추출 대상 문장 분류('sentence', 'script', 'narration')
        """
        limit = int(dataset[dataset['category']==category][f'n_{mode}'].mean())
        self.cum_ = np.zeros(limit)
        self.count_ = np.zeros(limit)
        def get_seqMean(seq) :
            crit = limit - len(seq)
            if crit > 0 :
                pad = [0]*crit
                seq += pad
            else :
                seq = seq[:limit]
            seq = np.array(seq)
            self.cum_ += np.array(seq)
            self.count_ += np.where(seq > 0, 1, 0)

        dataset[dataset['category']==category][f'{mode}LenSeq'].apply(get_seqMean)
        meanSeq = self.cum_/self.count_
        self.cum_ = None
        self.count_ = None        

        return meanSeq        