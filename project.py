# load the library
from bs4 import BeautifulSoup as Soup
import urllib, requests, re, pandas as pd

# indeed.com url
base_url = 'http://www.indeed.com/jobs?q=data+scientist&jt=fulltime&sort='
sort_by = 'date'          # sort by data
start_from = '&start='    # start page number

pd.set_option('max_colwidth',500)    # to remove column limit (Otherwise, we'll lose some info)
df = pd.DataFrame()   # create a new data frame
joblinks=[]
companylinks=[]
for page in range(0,10,1): # page from 1 to 100 (last page we can scrape is 100)
    page = page*10
    url = "%s%s%s%d" % (base_url, sort_by, start_from,page) # get full url 
    target = Soup(urllib.request.urlopen(url), "lxml")
    targetElements = target.find_all('div',{"class":" row result"}) # we're interested in each row (= each job)
    for result in targetElements:
        easyapply = result.find('span', attrs={'class':'iaLabel'})
        if easyapply !=None:
            home_url = "http://www.indeed.com"
            jobUrl = "%s%s" % (home_url,result.find('a').get('href'))
            jobUrl=jobUrl.replace("rc/clk", "viewjob")
            joblinks.append(jobUrl)

        else:
            continue
###############################Extract Features#########################################################
for link in joblinks:
    openJob=Soup(urllib.request.urlopen(link), "lxml")
    title=openJob.find('b',{'class':'jobtitle'}).getText()
    title=" ".join(title.split())
    company=openJob.find('span',{'class':'company'}).getText()
    company=" ".join(company.split())
    location=openJob.find('span',{'class':'location'}).getText()
    location=" ".join(location.split())
    description=openJob.find('span',{'class':'summary'}).getText()
    description=" ".join(description.split())
    df = df.append({'job_url': link, 'job_title': title, 
                        'company': company, 'location': location,
                        'description': description
                       }, ignore_index=True)

    
    

################################  RESUME PARSING   ##############################################################################

#import pip
#pip.main(["install","PyPDF2"])
import PyPDF2
import pandas as pd
df_resume = pd.DataFrame()

def find_between( s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            if last!="":
                return s[start:end]
            else:
                return s[start:-1]
        except ValueError:
            return ""            
file_to_read = '/home/harshal/Desktop/ResumeUpdated/input.txt'
if(file_to_read != ''):
    my_file = open(file_to_read)
    input_values = []
    for line in my_file:
        line=line.strip()
        input_values.append(line)
for resume in input_values:
    pdfFileObj = open(resume,'rb')     #'rb' for read binary mode
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    a=pdfReader.numPages
    print(a)
    pageObj = pdfReader.getPage(0)
    a=pageObj.extractText()
    a=a.replace('\n','')
    a=" ".join(a.split())
    print(a)
    summary=find_between( a, "SUMMARY", "EDUCATION" )
    education=find_between( a, "EDUCATION", "EXPERIENCE" )
    experience=find_between( a, "EXPERIENCE", "PROJECTS" )
    project=find_between( a, "PROJECTS", "Technical Skills" )
    skills=find_between( a, "Technical Skills", ""  )
    df_resume = df_resume.append({'summary': summary, 'education': education, 
                            'experience': experience, 'skills': skills, 
                            'project': project}, ignore_index=True)

############################################## TF-IDF ##############    
    ######### JOB TITLE SIMILARITY ##########################################

from stop_words import get_stop_words
stop_words_list = get_stop_words('en')

train_set = list(df['job_title'])
test_set = list(df_resume['summary'])

from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer()
#print(vectorizer)

vectorizer = CountVectorizer(stop_words=stop_words_list)
vectorizer.fit_transform(train_set)
print(vectorizer.vocabulary_)

job_title_matrix = vectorizer.transform(train_set)
summary_matrix = vectorizer.transform(test_set)
job_title_matrix.todense()
summary_matrix.todense()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(summary_matrix)
print("IDF:", tfidf.idf_)


tf_idf_job_title_matrix = tfidf.transform(job_title_matrix)
tf_idf_summary_matrix = tfidf.transform(summary_matrix)

#tf_idf_omatrix.todense()
#tf_idf_smatrix.todense()
from sklearn.metrics.pairwise import cosine_similarity
cosine_matrix_title = cosine_similarity(tf_idf_summary_matrix[0:1], tf_idf_job_title_matrix)
print(cosine_matrix_title)
cosine_matrix_title.argmax()

####################################################################################################
########################### Skills similarity ###########################

vectorizer_skills = CountVectorizer(stop_words = stop_words_list)
vectorizer_skills.fit_transform(df_resume['skills'])
print(vectorizer_skills.vocabulary_)


skills_matrix = vectorizer_skills.transform(df_resume['skills'])
skills_description_matrix = vectorizer_skills.transform(df['description'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_skills = TfidfTransformer(norm="l2")
tfidf_skills.fit(skills_description_matrix)

tf_idf_skills_matrix = tfidf_skills.transform(skills_matrix)
tf_idf_skills_description_matrix = tfidf_skills.transform(skills_description_matrix)

from sklearn.metrics.pairwise import cosine_similarity
cosine_matrix_skills = cosine_similarity(tf_idf_skills_matrix[0:1], tf_idf_skills_description_matrix)

cosine_matrix_skills
cosine_matrix_skills.argmax()




##############################################################################################

################################## Degree Similarity ################



degree_list = ['B.S' ,'BS', '4-year degree', 'MS','M.S','PHD', 'Ph.D','Bachelor','Bachelors','Associate','Masters','science']
find_degree = []
index_job_degree = []
dict_degree = {}
for deg in degree_list:
    for index in range(len(df['description'])):
        deg = deg.lower()
        descrip = df['description'][index].lower()
        if deg in  descrip:
#            find_degree.append(deg)
            
#dict_degree[index] = deg
            find_degree.append(deg)
            index_job_degree.append(index)

if 'ms' in df['description'][4]:
    print("hello")
            
degree = pd.read_csv('/home/harshal/Documents/SEMESTER  2/ML for Data Science/project/degree.csv',header  = None )

degree = list(degree[0])
degree_variations = ['B.S' ,'BS', '4-year degree', 'MS','M.S','PHD', 'Ph.D']

degree.extend(degree_variations)

vectorizer_degree = CountVectorizer(stop_words = stop_words_list)
vectorizer_degree.fit_transform(degree)
print(vectorizer_degree.vocabulary_)

degree_matrix = vectorizer_degree.transform(degree)
degree_resume_matrix = vectorizer_degree.transform(df_resume['education'])
degree_job_matrix = vectorizer_degree.transform(df['description'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_degree = TfidfTransformer(norm='l2')
tfidf_degree.fit(degree_matrix)

tf_idf_degree_resume_matrix = tfidf_degree.transform(degree_resume_matrix)
tf_idf_degree_job_matrix = tfidf_degree.transform(degree_job_matrix)

from sklearn.metrics.pairwise import cosine_similarity
cosine_matrix_degree = cosine_similarity(tf_idf_degree_resume_matrix[0:1], tf_idf_degree_job_matrix)

cosine_matrix_degree
cosine_matrix_degree.argmax()




















