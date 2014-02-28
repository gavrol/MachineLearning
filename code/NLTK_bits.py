'''
Created on 20/02/2014

@author: olena
'''


from nltk.book import *
import random


    
def grab_words(start_pos,text):
    count = random.randint(3,7)
    
    vec = text[start_pos:start_pos+count+1]
    title = []
    for elem in vec:
        if elem != ".":
            title.append(elem.lower())
            
    time_span = str(random.randint(5,60))+'min'
    if time_span == '5min': time_span = "lightning"
    title.append(time_span)
            
    title = ' '.join(title)
    return title, start_pos+count+1
    
            

NLTKbooks = [text1,text2,text3,text4,text5,text6,text7]
for txt in NLTKbooks: print txt

fn = open("PresentationTitles_v3.txt",'w')
generate_count = 120
text = text1

text = [word for word in text if word.isalpha() or word == '.']

fqmap = FreqDist(text)
vocab = fqmap.keys()
#print vocab[:100]

start_pos = text.index('.') + 1
count = 0
s = ""
while count < generate_count:
    title, start_pos = grab_words(start_pos,text)
    s += title+"\n"
    count += 1

fn.write(s)
fn.close()



