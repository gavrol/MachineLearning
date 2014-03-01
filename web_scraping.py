'''
Created on 01/03/2014

@author: olena
'''
import httplib2
import json
from nltk import clean_html
#from bs4 import BeautifulSoup,BeautifulStoneSoup
from BeautifulSoup import BeautifulStoneSoup
import apiclient.discovery # pip install google-api-python-client
API_KEY = 'AIzaSyCcoaSseZNZLRR11nJFxgcurSevsruYS98'
# XXX: Enter any person's name
Q = "Tim O'Reilly" #"Sven Krumke"

def cleanHtml(html):
    if html == "": return ""
    return BeautifulStoneSoup(clean_html(html),
                              convertEntities=BeautifulStoneSoup.HTML_ENTITIES).contents[0]
service = apiclient.discovery.build('plus', 'v1', http=httplib2.Http(), developerKey=API_KEY)
people_feed = service.people().search(query=Q).execute()
print type(people_feed) #should be dict
print people_feed.keys() #[u'nextPageToken', u'kind', u'title', u'items', u'etag', u'selfLink']
#the only item in people_feed worth exploring is 'items'

recs = people_feed['items'] # recs is a list, what are the elements of that list
print type(recs[0]) # should be dict with these keys: [u'kind', u'displayName', u'url', u'image', u'etag', u'id', u'objectType']
#for rec in recs: print rec['kind'],"\n",rec['displayName'],"\n\n"
#print json.dumps(people_feed['items'], indent=1)

IDs = [] 
for rec in recs: IDs.append(rec['id'])
print IDs

#USER_ID = '107033731246200681024' # Tim O'Reilly

Activities = []
for Id in IDs:
    activity_feed = service.activities().list( userId=Id, collection='public',maxResults='100').execute()
    Activities.append(activity_feed)
    
type(Activities[0]) #<type 'dict'>
print Activities[0].keys() #[u'nextPageToken', u'kind', u'title', u'items', u'updated', u'etag']

type(Activities[0]['items']) #<type 'list'>
print Activities[0]['items'] #a long-long thing

acts = Activities[0]['items']
print acts[0]
print acts[0].keys() #[u'kind', u'provider', u'title', u'url', u'object', u'updated', u'actor', u'access', u'verb', u'etag', u'published', u'id']
print acts[0]['title'] 

print cleanHtml(acts[0]['object']['content'])
                           
