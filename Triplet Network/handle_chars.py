# -*- coding: utf-8 -*-

import os
import re


def _is_invalid_line(line_txt):
    """
    function:check the current sentence is invalid
    line_txt:the line content which is in string format
    """

    ##1. The blank line
    if line_txt == '':
        return True
    #line_content = re.sub(r'[^A-Za-z0-9\'!\"#$%&\'()*\+,-.:;<=>?@\[\]_{|}\s]', '', line_txt).decode('utf8','ignore').strip()

    tmp = re.sub(r'[^A-Za-z0-9\s]', '', line_txt)
    #2. if the length of a sentence except only contain digit, punctuation or special character is lower than 3, it should be a invalid sentence
    if tmp.isdigit() or len(tmp) <= 3:
        return True

    #3. handle like (cid:180)
    if len(re.sub("(cid\d*)+", '', tmp))<3:
        return True

    # what out: all blank spaces have been replaced
    ## 4. H K 2 0 1 4
    if re.search("HK\d+", tmp) and len(set([len(e) for e in line_txt.split()]))  <3:
        return True

    if len(line_txt.split())==1:   ### dhsjdhjskdhjsdjs
        return True
    ## 5. Million Million Million ... or Hong Kong Hong Kong ...
    if len(set(tmp.split()))<3:
        return True

    return False


def _is_title(line_txt):
    """
    function: check whether the current line is title or directory
    line_txt: the target line content in string format
    """
    conj_words = ['or', 'and', 'at', 'in', 'by', 'with', 'of']
    
    # initial the variable
    words = line_txt.split()
    is_title = True

    # all words' Capital letter of a title except conj should be capitalized 
    for word in words:
        if word not in conj_words and not word[0].isupper():
            is_title = False

    return is_title


def deal_with_sentence(line_txt):
    """
    function: deal with single line 
    line_txt: the target line in string format
    """
        
    ## filter all special characters
    ## leave alpha, digit, '.' and ','
    ## but '?)"'may be a line end sign ?
    #line_content = re.sub(r"[^A-Za-z0-9,.’]", ' ', line_txt) \
        #.decode("utf-8", 'ignore').strip()
    init_content=line_txt 
    line_content=re.sub(r'[^A-Za-z0-9\'!\"#$%&\'()*\+,-.:;<=>?@\[\]_{|}\s]', '', line_txt).decode('utf8', 'ignore').strip()
    line_content=re.sub('\(Cid:\d*\)', '', line_content,flags=re.IGNORECASE)
    line_content=re.sub('\s+', ' ', line_content)
    # check whether the line is a title
    if _is_title(init_content.strip()):
        line_content += '\n'
    else:
        # check whether the line is a complete sentence
        if line_content[-1] == '.':
            line_content += '\n'
        
        else:   # if the line is not a compete sentence, it should be a part of behine content, so add a blank space
            line_content += ' '
    return line_content


def split_chunk(txt_path, txt_name, target_path):
    """
    function: split the txt contnet into many chuncks
    txt_path: the target txt path
    txt_name: the target txt name
    """
    #print 'start to split chunk ----------------'
    result_content = ''  ## the final result content
    chunk_content = ''
    
    with open(txt_path + txt_name, "rb") as txt_f:
        lines=txt_f.readlines()
        flag=False
        for line in lines:
            flag=True
            #result_content = re.sub(r'[^A-Za-z\s]', '', line)
            if _is_invalid_line(line.strip()): ## if the sentence is invalid, it is a end sign of chunk, and the chunk should be writen in result content
                if chunk_content != '':
                    result_content += chunk_content
                    flag=False
                    result_content += '\r\n'  ### use '\n' to separate chunks
                    chunk_content = ''   ## start a new chunk

            else:   ### add the sentence into the chunk if it is valid
                line = deal_with_sentence(line)
                chunk_content += line
                flag=True

        if flag:
            result_content +=chunk_content
    if len(result_content.strip())>0:
        with open(target_path+txt_name[:10], "w") as txt_f:
            txt_f.write(result_content.lower())
        txt_f.close()


def process(source_dir, target_dir):
    file_list = os.listdir(source_dir)
    for txt_name in file_list:
        
        if not re.search('.txt$', txt_name):
            continue
        
        split_chunk(source_dir, txt_name, target_dir)
        

if __name__ == '__main__':
   
    txt_path_in='talk.politics.misc/'
    target_dir='talk_politics_misc/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    process(txt_path_in, target_dir)

