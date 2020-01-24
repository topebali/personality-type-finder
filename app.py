#import all the required python Packages

from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory, send_file
import pandas as pd
import numpy as np
import json 
import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import re



#create a flask Applicaton
app = Flask(__name__)

class DataStore():
    data1=None
    datadisplay=None
data=DataStore()

#Create an application route
@app.route("/",methods=["GET","POST"])


def homepage():
    #Call value from HTML form
    data1 = request.form.get('question1_field', 'Education')
    params = data1
    data.data1 =data1

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.options.mode.chained_assignment = None

    #Read raw data as a dataframe
    csv_file = r"C:\Users\Tope Bali\Documents\Python Scripts\personality-type-finder\Personalities.csv"
    datasource1 = pd.read_csv(csv_file)
    datasource1 = pd.DataFrame(datasource1)
    #datasource1 = datasource1.values.ravel().tolist()

    #Seperate words into multiple entries

    #words = words.split() #(This wil be stored as seperate items in a list)
    words = data1.split() #(This wil be stored as seperate items in a list)
    


    #filter the data for posts that contain either of these words
    datasource2 = []

    for i in words:
        d2 = datasource1[datasource1['posts'].str.contains(str(i))]
        #d2 = datasource1[datasource1['posts'].str.contains(str(i), "(?i)")]

        datasource2.append(d2)

    #datasource = pd.concat(datasource, ignore_index=True)
    datasource = pd.concat(datasource2, ignore_index=True)

    datasource2 = pd.DataFrame(datasource2)
    datasource2 = datasource.drop_duplicates()
    datasource2 = datasource.reset_index()
    
    
    datasource = datasource2
    index_source = 0
    max_simi = 0

    #documents = []
    for i in range (0, len(datasource)):
        documents = []
        documents.append(params)
        documents.append(datasource.loc[i, 'posts'])
        #print(documents)

        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 4)).fit_transform(documents)


        #compute pairwise similarity by multiplying by transpose.
        pairwise_similarity = tfidf * tfidf.T

        #Then, extract the score by selecting the first element in the matrix.
        simiscore = pairwise_similarity.A[0][1]

        #format the similarity score to select only 4 decimals
        simiscore = "{0:.4f}".format(simiscore)

        #finally, add the similarity score to the Df
        datasource.loc[i, 'Simiscore'] = simiscore

    #compute th rank and add it to th dataframe as its own column
    datasource['Rank'] = datasource['Simiscore'].rank(method='average', ascending=False)
    datasource['Search Term']=data1

    #sort by the rank
    datasource = datasource.sort_values('Rank', ascending=True)

    #Select top 5 entries 
    datasource = datasource.head(5)
    datasource = datasource.reset_index()

    #Add the search term to the dataframe as its own column
    #datasource['Search Term'] = pd.Series(words)


    #Select relevant columns and assign to a dataframe called docs
    
    docs = datasource[["type" ,"Rank","Simiscore","Search Term","Introversion/Extraversion","Intuitive/Observant","Thinking/Feeling","Judging/Perceiving"]]
    
    
    docs.Rank = docs.Rank.astype(np.int64)
    #Convert to json 
    docs = docs.to_json(orient="records")

    #return data to front end for display 
    #return render_template('index.html', docs=json.loads(docs))
    return render_template('index.html', docs=json.loads(docs),data1=data1)



if __name__ == "__main__":
    app.run(debug=True)

