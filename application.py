import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import io
import base64
import pandas as pd
import dash_table
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.stem import SnowballStemmer
import plotly.express as px
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, naive_bayes
import plotly.graph_objects as go
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis.gensim
import os
import dash_auth

#-------


#-------



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__)
dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash_app.server


dash_app.config.suppress_callback_exceptions = True

#Authetication
VALID_USERNAME_PASSWORD_PAIRS = {
    'Text': 'Kabra'
}



#dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#auth = dash_auth.BasicAuth(dash_app, VALID_USERNAME_PASSWORD_PAIRS)

#app = dash.Dash('auth')
auth = dash_auth.BasicAuth(
    dash_app,
    (('ak1','1234',),)
)

# Authetication


dash_app.layout = html.Div([
    html.Div(" "),
    html.Br(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


image_filename1 = 'FakeRibbon.PNG'
image_filename2 = 'FakeRibbon1.PNG'

##encoded_image1 = base64.b64encode(open(image_filename1, 'rb').read())

#encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())


index_page = html.Div([
     html.H1('ERM ', style={'background-image': 'url(https://erm.com/)','color': 'green','size':'30'})
,
    html.Img(src = dash_app.get_asset_url(image_filename1)),
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image1.decode())),
  #  html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode())),
    dcc.Markdown('''
The digitalization of EHSS: Approaching a tipping point

Technology has been part of the environmental, health, safety and sustainability (EHSS) landscape for some time. However, over the past few years, we’ve seen this landscape revolutionized by digitalization and the industrialization and widespread commercialization of technology.
Digital pioneers such as Equinor, United Airlines and formerly DowDuPont, who embraced innovative technologies early on, are already citing benefits that stretch well beyond EHSS metrics and are actually driving operational results. 


These improvements come not from incremental changes but from the radical transformation of EHSS, which can help to reduce risk profiles at an enterprise-level and change the day-to-day roles of frontline workers and EHSS staff.
The adoption of innovative technologies and solutions is not only transforming how organizations tackle their EHSS challenges but is also driving greater efficiency and intelligence. Technology, especially the availability of data and the accessibility of analytics and visualization tools, is enabling leading companies to move from hindsight to insight and in some instances, foresight.
''', style={'background-image': html.Img(src = dash_app.get_asset_url(image_filename1))})  ,
    dcc.Link('Safety Observations Classification and Theme identification', href='/Observations' ,
             style={'background-image': html.Img(src = dash_app.get_asset_url(image_filename1))}),
    html.Br(),
   
     html.Br(),
    dcc.Link('Incident Injury Keyword extraction and Theme identification', href='/Incidents'),
    html.Br(),

])

#-------

#function to combine the text data
def create_text_data(description_column): 
    description_data = ""
    for i in description_column:
        description_data += " "+i
    return description_data

#Function to draw the wordcloud
def draw_wordcloud(description_data):
    wordcloud_raw_data = WordCloud().generate(description_data)
    #plt.imshow(wordcloud_raw_data)
    #plt.axis("off")
    #plt.margins(x=0,y=0)
    #plt.title(title)
    #plt.show()
    return wordcloud_raw_data

def remove_stopwords(text_col):
    #data["NO_STOPWORDS_DESCRIPTION"] = "0"  #column for creating description without stopwords
    no_sw_text = []
    for i in range(0,len(text_col)):
        text = text_col[i].lower()
        tokenized_words = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] 
        tokens_no_stop_words = [word for word in tokenized_words if word not in stopwords.words("english")]
        no_stopwords_text = " ".join([j for j in tokens_no_stop_words])
        no_sw_text.append(no_stopwords_text)
        
    return no_sw_text 


def data_classification():
    preprocess_data = pd.read_csv("Usecase1_Upload.csv")
    #preprocess_data = preprocessing_data(raw_data)
    col = ['INJPOT',"ACTCATEGORY","Safe_Unsafe_Category", 'NO_STOPWORDS_DESCRIPTION']  #Subsetting dataframe on the basis of two columns 
    data = preprocess_data[col]
    data = data[pd.notnull(data['NO_STOPWORDS_DESCRIPTION'])]  #Removing the row which does not have description
    data.columns = ['INJPOT',"ACTCATEGORY","Safe_Unsafe_Category", 'NO_STOPWORDS_DESCRIPTION']  #Naming the columns of new dataframe 

    #Converting INJPOT to a catgeory id for classification
    data['INJPOT_category_id'] = data['INJPOT'].factorize()[0]  #Converting the ACTTYPE to factors so that it contains the values of 0, 1, 2 etc
    INJPOT_category_id_df = data[['INJPOT', 'INJPOT_category_id']].drop_duplicates().sort_values('INJPOT_category_id')  #Fetching the uniques factorize values
    INJPOT_category_to_id = dict(INJPOT_category_id_df.values)  #Creating a mapping of acttype to category id
    id_to_INJPOT_category = dict(INJPOT_category_id_df[['INJPOT_category_id', 'INJPOT']].values)
    

    #Converting ACTCATEGORY to a catgeory id for classification
    data['ACTCATEGORY_category_id'] = data['ACTCATEGORY'].factorize()[0]  #Converting the ACTTYPE to factors so that it contains the values of 0, 1, 2 etc
    ACTCATEGORY_category_id_df = data[['ACTCATEGORY', 'ACTCATEGORY_category_id']].drop_duplicates().sort_values('ACTCATEGORY_category_id')  #Fetching the uniques factorize values
    ACTCATEGORY_category_to_id = dict(ACTCATEGORY_category_id_df.values)  #Creating a mapping of acttype to category id
    id_to_ACTCATEGORY_category = dict(ACTCATEGORY_category_id_df[['ACTCATEGORY_category_id', 'ACTCATEGORY']].values)
   
   

    #Converting Safe Unsafe Category to a catgeory id for classification
    data['Safe_Unsafe_Category_id'] = data['Safe_Unsafe_Category'].factorize()[0]  #Converting the ACTTYPE to factors so that it contains the values of 0, 1, 2 etc
    Safe_Unsafe_category_id_df = data[['Safe_Unsafe_Category', 'Safe_Unsafe_Category_id']].drop_duplicates().sort_values('Safe_Unsafe_Category_id')  #Fetching the uniques factorize values
    Safe_Unsafe_category_to_id = dict(Safe_Unsafe_category_id_df.values)  #Creating a mapping of acttype to category id
    id_to_Safe_Unsafe_category = dict(Safe_Unsafe_category_id_df[['Safe_Unsafe_Category_id', 'Safe_Unsafe_Category']].values)
    
    
    #Running the Classification on Injpot 

    train_x = data['NO_STOPWORDS_DESCRIPTION']
    valid_x = data['NO_STOPWORDS_DESCRIPTION']
    train_y = data['INJPOT_category_id']
    valid_y = data['INJPOT_category_id']

    encoder = preprocessing.LabelEncoder()

    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer TF-DTM object 
    count_vect = CountVectorizer(analyzer='word')
    #t1 = time.time()
    count_vect.fit(data['NO_STOPWORDS_DESCRIPTION'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    #t2 = time.time()

    #print(t2 - t1, "secs")  # ~ 0.15 secs to create DTM. Not bad, eh?
    xtrain_count.shape
    xvalid_count.shape

    MNB = naive_bayes.MultinomialNB().fit(xtrain_count, train_y)

    # predict the labels on validation dataset
    #predictions = naive_bayes.MultinomialNB().predict(xvalid_count)

    predictions = MNB.predict(xvalid_count)

    #print(len(predictions))
    predDF = pd.DataFrame()
    predDF['ID'] = list(range(1,len(valid_x)+1))
    predDF['Description'] = valid_x
    predDF['INJPOT_actual_label'] = valid_y
    predDF['INJPOT_model_label'] = predictions
    
    #Running the Classification on ACTCATEGORY 

    train_y = data['ACTCATEGORY_category_id']
    valid_y = data['ACTCATEGORY_category_id']

    encoder = preprocessing.LabelEncoder()

    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer TF-DTM object 
    count_vect = CountVectorizer(analyzer='word')
    #t1 = time.time()
    count_vect.fit(data['NO_STOPWORDS_DESCRIPTION'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    #t2 = time.time()

    #print(t2 - t1, "secs")  # ~ 0.15 secs to create DTM. Not bad, eh?
    xtrain_count.shape
    xvalid_count.shape

    MNB = naive_bayes.MultinomialNB().fit(xtrain_count, train_y)

    # predict the labels on validation dataset
    #predictions = naive_bayes.MultinomialNB().predict(xvalid_count)

    predictions = MNB.predict(xvalid_count)

    #print(len(predictions))
    #predDF = pd.DataFrame()
    #predDF['text'] = valid_x
    predDF['ACTCATEGORY_actual_label'] = valid_y
    predDF['ACTCATEGORY_model_label'] = predictions

    #Running the Classification on Safe_Unsafe_Category 

    train_y = data['Safe_Unsafe_Category_id']
    valid_y = data['Safe_Unsafe_Category_id']

    encoder = preprocessing.LabelEncoder()

    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer TF-DTM object 
    count_vect = CountVectorizer(analyzer='word')
    #t1 = time.time()
    count_vect.fit(data['NO_STOPWORDS_DESCRIPTION'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    #t2 = time.time()

    #print(t2 - t1, "secs")  # ~ 0.15 secs to create DTM. Not bad, eh?
    xtrain_count.shape
    xvalid_count.shape

    MNB = naive_bayes.MultinomialNB().fit(xtrain_count, train_y)

    # predict the labels on validation dataset
    #predictions = naive_bayes.MultinomialNB().predict(xvalid_count)

    predictions = MNB.predict(xvalid_count)

    #print(len(predictions))
    #predDF = pd.DataFrame()
    #predDF['text'] = valid_x
    predDF['SAFE_UNSAFE_actual_label'] = valid_y
    predDF['SAFE_UNSAFE_model_label'] = predictions
    
    inj_act_label = predDF["INJPOT_actual_label"].unique()
    act_actual_label = predDF["ACTCATEGORY_actual_label"].unique()
    safe_unsafe_label = predDF["SAFE_UNSAFE_actual_label"].unique()

    inj_act_label
    act_actual_label
    safe_unsafe_label

    id_to_ACTCATEGORY_category
    id_to_INJPOT_category
    x = list(id_to_Safe_Unsafe_category.values())
    #x[safe_unsafe_label[0]]
    str(safe_unsafe_label[0])

    safe_unsafe_dict = {}
    actcategory_dict = {}
    injpot_dict = {}

    Safe_Unsafe_Category_values = list(id_to_Safe_Unsafe_category.values())
    Injpot_category_values = list(id_to_INJPOT_category.values())
    Act_category_values = list(id_to_ACTCATEGORY_category.values())


    for i in safe_unsafe_label:
        safe_unsafe_dict[str(i)] = Safe_Unsafe_Category_values[i]

    safe_unsafe_dict

    for i in inj_act_label:
        injpot_dict[str(i)] = Injpot_category_values[i]



    for i in act_actual_label:
        actcategory_dict[str(i)] = Act_category_values[i]

    actcategory_dict
    injpot_dict
    safe_unsafe_dict
    
    injpot_model_text = []
    injpot_model = list(predDF["INJPOT_model_label"])
    for i in injpot_model:
        text = str(injpot_dict.get(str(i)))
        if text == 'nan':
            text = "Blank"
        injpot_model_text.append(text)

    injpot_model_text

    actcategory_model_text = []
    actcat_model = list(predDF["ACTCATEGORY_model_label"])
    for i in actcat_model:
        text = str(actcategory_dict.get(str(i)))
        if text == 'nan':
            text = "Blank"
        actcategory_model_text.append(text)

    actcategory_model_text

    safe_unsafe_model_text = []
    safe_unsafe_model = list(predDF["SAFE_UNSAFE_model_label"])
    for i in safe_unsafe_model:
        text = str(safe_unsafe_dict.get(str(i)))
        if text == 'nan':
            text = "Blank"
        safe_unsafe_model_text.append(text)

    safe_unsafe_model_text
    
    injpot_actual_text = []
    injpot_actual = list(predDF["INJPOT_actual_label"])
    for i in injpot_actual:
        text = str(injpot_dict.get(str(i)))
        if text == 'nan':
            text = "Blank"
        injpot_actual_text.append(text)

    injpot_actual_text

    actcategory_actual_text = []
    actcat_actual = list(predDF["ACTCATEGORY_actual_label"])
    for i in actcat_actual:
        text = str(actcategory_dict.get(str(i)))
        if text == 'nan':
            text = "Blank"
        actcategory_actual_text.append(text)

    actcategory_actual_text

    safe_unsafe_actual_text = []
    safe_unsafe_actual = list(predDF["SAFE_UNSAFE_actual_label"])
    for i in safe_unsafe_actual:
        text = str(safe_unsafe_dict.get(str(i)))
        if text == 'nan':
            text = "Blank"
        safe_unsafe_actual_text.append(text)

    safe_unsafe_actual_text

    predDF["INJPOT_Predicted_Text"] = injpot_model_text
    predDF["ACTCATEGORY_Predicted_Text"] = actcategory_model_text
    predDF["SAFE_UNSAFE_Predicted_Text"] = safe_unsafe_model_text
    predDF["INJPOT_Actual_Text"] = injpot_actual_text
    predDF["ACTCATEGORY_Actual_Text"] = actcategory_actual_text
    predDF["SAFE_UNSAFE_Actual_Text"] = safe_unsafe_actual_text
    predDF.to_csv("All_Classifications.csv")
    
    ## Code for visualizing confusion matrix
    #predDF["confusion_code"] = "0"
    #predDF["Actual_Label_Text"] = "0"
    #predDF["Predicted_Label_Text"] = "0"
    
    if False: 
    
        serious_word_dictionary = ["hand","leg","goggle","glass","gloves","three point","material"]

        stemmer = SnowballStemmer('english')
        def string_matcher(dict_str,sent):
            result = 0
            #print dict_str
            #print sent


            if dict_str in sent:
                dict_str_split = dict_str.split(" ")
                dict_str_len = len(dict_str_split)
                sent_split = sent.split(" ")
                #print sent_split
                words_sent = len(sent_split)+1
                for i in range(0,words_sent-dict_str_len):
                    ctr = 0
                    #print i
                    for j in range(i,dict_str_len+i):
                        clean_sent = sent_split[j].replace(",","")
                        clean_sent = clean_sent.lower()
                        dict_str_split[ctr] = dict_str_split[ctr].lower()
                        if stemmer.stem(dict_str_split[ctr]) == stemmer.stem(clean_sent):
                            ctr = ctr+1
                        else: 
                            break

                    #print ctr
                    if ctr==dict_str_len:
                        result = 1
            return result

        Y1 = predDF.actual_label==1
        Y2 = predDF.model_label == 0
        Y = Y1 & Y2
        serious_minor_data = predDF[Y]
        indexes = serious_minor_data.index

        for i in range(0,serious_minor_data.shape[0]):
            text = serious_minor_data["text"].iloc[i]
            tokenized_words = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            stemmed_tokens_no_sw = [stemmer.stem(word) for word in tokenized_words]
            stemmed_text = " ".join([j for j in stemmed_tokens_no_sw])
            injury_str = ""
            for j in serious_word_dictionary:
                #print(j)
                tokenized_words_j = [word for sent in nltk.sent_tokenize(j) for word in nltk.word_tokenize(sent)]
                stem_j = [stemmer.stem(word) for word in tokenized_words_j]
                j_text = " ".join([j for j in stem_j])
                valid = string_matcher(j_text.lower(),stemmed_text.lower())
                if valid ==1:
                    predDF["model_label"].loc[indexes[i]] = 1

        for i in range(0,predDF.shape[0]):
            act = predDF["actual_label"].iloc[i]
            mod = predDF["model_label"].iloc[i]
            predDF["Actual_Label_Text"].iloc[i] = id_to_category.get(act)
            predDF["Predicted_Label_Text"].iloc[i] = id_to_category.get(mod)

            if (act==mod and act==1):
                predDF["confusion_code"].iloc[i] = "X11"
            elif (act==mod and act==0):
                predDF["confusion_code"].iloc[i] = "X00"
            elif (act!=mod and act == 0):
                predDF["confusion_code"].iloc[i] = "X01"
            else:
                predDF["confusion_code"].iloc[i] = "X10"

        predDF.to_csv("Pred_DF_v1.csv")
        X_00_count = predDF[predDF["confusion_code"]=="X00"].shape[0]
        X_01_count = predDF[predDF["confusion_code"]=="X01"].shape[0]
        X_10_count = predDF[predDF["confusion_code"]=="X10"].shape[0]
        X_11_count = predDF[predDF["confusion_code"]=="X11"].shape[0]

        



        confusion_matrix = [[X_00_count,X_01_count],[X_10_count,X_11_count]]
    return "Success"
#html.Div(html.H1("Hello"))

    
def parse_contents_uc1(contents,filename):
    #print(contents)
    #print(filename)
    print("parse contents code")
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            raw_data = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            #Step 1: Adding safe and unsafe column
            safe_unsafe_list = []
            col = raw_data.ACTTYPE
            for i in col:
                if i[0]=="U":
                    safe_unsafe_list.append("Unsafe")
                elif i[0]=="S":
                    safe_unsafe_list.append("Safe")
                else:
                    safe_unsafe_list.append("Wrong_Data")
            safe_unsafe_list
            raw_data["Safe_Unsafe_Category"] = safe_unsafe_list
            preprocess_data = raw_data
            preprocess_data["NO_STOPWORDS_DESCRIPTION"] = remove_stopwords(preprocess_data["DESCRIPTION"])
    
            preprocess_data.to_csv("Usecase1_Upload.csv")
            #no_sw_data = create_text_data(preprocess_data.NO_STOPWORDS_DESCRIPTION_CACHED)
                      
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            raw_data = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        #html.Div(children=draw_wordcloud(no_sw_data,"sample_data")),
        dash_table.DataTable(
            data=raw_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in raw_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
            style_cell={'textAlign': 'left'}
            
        )
    ])




dash_app.layout = html.Div([
    html.Div("Page showing multiple options to navigate to different pages"),
    html.Br(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])



page_1_layout = html.Div([
                    html.Div(children = [
                    #    dcc.Link('Incident Injury Keyword extraction and Theme identification', href='/Incidents'),
                        html.H1('Safety Observations Classification and Theme identification ', style={'background-image': 'url(https://erm.com/)','color': 'green','size':'30'}),
                        
                        html.Img(src = dash_app.get_asset_url(image_filename1)),
                        html.Br(),
                        html.Br(),
                        dcc.Link('Main Page', href='/'),
                        html.Br(),
                        html.Br(),
                        dcc.Upload(html.Button("Upload File")),
                        dcc.Upload(id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            }), 
                        html.Div(id='output-data-upload'), 
                        html.Hr(),
                        html.Button("Visualization",id="vis_uc1"),
                        html.Div(id ="Wordcloud"),
                        html.Br(),
                        html.Button("Classification",id="class_vis_uc1"),
                        dcc.Graph(id="Graph 1",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
                        dcc.Graph(id="Graph 2",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
                        dcc.Graph(id="Graph 3",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
                        html.Div(id='container-button-basic'),
                        dcc.Dropdown(
                                    id='my-dropdown',
                                    options=[
                                        {'label': 'ACTCATEGORY', 'value': 'ACTCAT'},
                                        {'label': 'SAFE UNSAFE', 'value': 'SAFEUNSAFE'},
                                        {'label': 'INJURY POTENTIAL', 'value': 'INJPOTCAT'}
                                    ]

                                ),
                                #html.Div(id = "output-container")
                                dcc.Graph(id="Confusion Matrix",figure={'layout': {
                                                               'clickmode': 'event+select'}},style={'display':'none'}),
                        html.Div(id="confusion_matrix_observation"),
                        html.Button('Topic Modelling', id='UC1_topic_modelling'),
                        html.Div(id="pyLDAvis_Output"),
                        html.Br(),
                        html.Br(),
                        html.Button('Topic Distribution',id="UC1_Topic_Distribution"),
                        dcc.Graph(id = "UC1_Topic_Distribution_Output",figure= {'layout': {
                                                        'clickmode': 'event+select'}},style = {'display':'none'}),
                        html.Div(id= "UC1_Observations")
                                                    ])
    
])


### Callback for uploading the data and showing the contents of the file in webpage
@dash_app.callback(Output('output-data-upload','children'),
              [Input('upload-data','contents')],
              [State('upload-data','filename')])

def Update_output(list_of_contents,list_of_names):
    print("showing the contents of the file")
    if list_of_contents is not None: 
        print("Inside list of contents")
        children = [
            parse_contents_uc1(list_of_contents, list_of_names) ]
        return children 

@dash_app.callback(Output('Wordcloud','children'),
             [Input('vis_uc1','n_clicks')])

def show_wordcloud(n_clicks):
    print("show wordcloud function")
    if n_clicks>0:

        preprocess_data = pd.read_csv("Usecase1_Upload.csv")
        print("Pre-processing done")
        description_text = create_text_data(preprocess_data["NO_STOPWORDS_DESCRIPTION"])
        print("Description Text done")
        wc = draw_wordcloud(description_text)
        wc.to_file("Original_Data_Wordcloud.png")
        print("File Saving DOne.")
        image_filename = 'Original_Data_Wordcloud.png' # replace with your own image
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())


        return html.Div([html.H1("How data looks like!!"),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode("utf-8"))) 
                        ])
        
    
@dash_app.callback(dash.dependencies.Output('Graph 1', 'style'),
    [dash.dependencies.Input('class_vis_uc1', 'n_clicks')])

def show_graph_1(n_clicks):
    #print("show graph 1")
    #print(n_clicks)
    if n_clicks>0:
        indicator = data_classification()
        if indicator == "Success":
            return {'display':'block'}
    else: 
        return {'display':'none'}

    
### Callback for process data button   
@dash_app.callback(Output('Graph 1','figure'),
             [Input('class_vis_uc1','n_clicks')])

def Graph_1_processing(n_clicks):
    if n_clicks>0:
        print("Graph 1 Processing ")
        #indicator = data_classification()
        #if indicator == "Success":
        print("Data classification Success")
        all_class = pd.read_csv("All_Classifications.csv")
        unique_count = all_class.groupby("ACTCATEGORY_Predicted_Text")["ID"].nunique()
        #unique_count
        xlabel = unique_count.index

        #xlabel = []
        #for i in all_class["ACTCATEGORY_model_label"].unique():
         #   xlabel.append(actcategory_dict.get(str(i)))

        return {'data':[go.Bar(
                    x=xlabel,y=unique_count)],
            'layout': {
                'clickmode': 'event+select',

            }
        }


        
@dash_app.callback(
    dash.dependencies.Output('Graph 2', 'style'),
    [dash.dependencies.Input('Graph 1', 'clickData')])

def show_graph2(clickData):
    if clickData is not None: 
        print("show graph 2")
        print(clickData)
        return {'display':'block'}
    else:
        return {'display':'none'}
        
@dash_app.callback(
    dash.dependencies.Output('Graph 2', 'figure'),
    [dash.dependencies.Input('Graph 1', 'clickData')])

def Graph_2_processing(clickData):
     
    print("graph 2 processing")
    X = clickData
    Y = X["points"][0]
    print(Y)
    x_coordinate = Y['x']
    #y_coordinate = Y['y']
    print(x_coordinate)

    all_class = pd.read_csv("All_Classifications.csv")

    first_level_class = all_class[all_class["ACTCATEGORY_Predicted_Text"]==str(x_coordinate)]

    first_level_class.to_csv("First_Level_Classification.csv")

    fc_unique_count = first_level_class.groupby("SAFE_UNSAFE_Predicted_Text")['ID'].nunique()
    fc_unique_count

    fc_xlabel = fc_unique_count.index

    return {'data':[go.Bar(
                    x=fc_xlabel,y=fc_unique_count)],
            'layout': {
                'clickmode': 'event+select',

            }
        }
    
    

@dash_app.callback(
    dash.dependencies.Output('Graph 3', 'style'),
    [dash.dependencies.Input('Graph 2', 'clickData')])

def show_graph3(clickData):
    
    if clickData is not None:
        print("show graph3")
        return {'display':'block'}
    else:
        return {'display':'none'}            

@dash_app.callback(
    dash.dependencies.Output('Graph 3', 'figure'),
    [dash.dependencies.Input('Graph 2', 'clickData')])

def Graph_3_processing(clickData):
    
    X = clickData
    Y = X["points"][0]
    print(Y)
    x_coordinate = Y['x']
    #y_coordinate = Y['y']
    print(x_coordinate)
    #print(y_coordinate)
    #print(X)

    first_level_class = pd.read_csv("First_Level_Classification.csv")

    second_level_class = first_level_class[first_level_class["SAFE_UNSAFE_Predicted_Text"]==str(x_coordinate)]

    second_level_class.to_csv("Second_Level_Classification.csv")

    sc_unique_count = second_level_class.groupby("INJPOT_Predicted_Text")['ID'].nunique()
    sc_unique_count

    sc_xlabel = sc_unique_count.index

    return {'data':[go.Bar(
                    x=sc_xlabel,y=sc_unique_count)],
            'layout': {
                'clickmode': 'event+select',

            }
        }

@dash_app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('Graph 3', 'clickData')])    
    
def print_wordcloud(clickData):
    
    X = clickData
    Y = X["points"][0]
    print(Y)
    x_coordinate = Y['x']
    #y_coordinate = Y['y']
    print(x_coordinate)
    #print(y_coordinate)
    #print(X)

    second_level_class = pd.read_csv("Second_Level_Classification.csv")

    third_level_class = second_level_class[second_level_class["INJPOT_Predicted_Text"]==str(x_coordinate)]

    third_level_class.to_csv("Third_Level_Classification.csv")

    description_text = create_text_data(third_level_class["Description"])

    wc = draw_wordcloud(description_text)

    wc.to_file("Final_Classification_Wordcloud.png")

    image_filename = 'Final_Classification_Wordcloud.png' # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#encoded_image.

#print(encoded_image))

#y = 'data:image/png;base64,{}'.format(encoded_image.decode("utf-8"))
    return  html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode("utf-8"))),
        dash_table.DataTable(
            data=third_level_class.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in third_level_class.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
            style_cell={'textAlign': 'left'}
        )

    ])

@dash_app.callback(
    dash.dependencies.Output('Confusion Matrix', 'style'),
    [dash.dependencies.Input('my-dropdown', 'value')])

def show_confusion_matrix(value):
    print("Show confusion matrix function")
    if value is not None: 
        print("Show if conditon passed")
        return {'display':'block'}
    else: 
        return {'display':'none'}

@dash_app.callback(
    dash.dependencies.Output('Confusion Matrix', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')])

def create_confusion_matrix(value):
    print("Create confusion matrix functions")
 
    print("Create if condition passed")
    all_class = pd.read_csv("All_Classifications.csv")
    if value == "ACTCAT":
        actual_label = all_class["ACTCATEGORY_Actual_Text"].unique()
        predicted_label = all_class["ACTCATEGORY_Predicted_Text"].unique() 
    elif value == "SAFEUNSAFE":
        actual_label = all_class["SAFE_UNSAFE_Actual_Text"].unique()
        predicted_label = all_class["SAFE_UNSAFE_Predicted_Text"].unique()
    elif value == "INJPOTCAT":
        actual_label = all_class["INJPOT_Actual_Text"].unique()
        predicted_label = all_class["INJPOT_Predicted_Text"].unique()

    confusion_matrix = []
    diag_valid_mat = []
    non_diag_records = 0
    for i in actual_label:
        row_mat = []
        row_diag_mat = []
        for j in predicted_label:
            
            if value == "ACTCAT":
                Y1 = all_class["ACTCATEGORY_Actual_Text"]==i
                Y2 = all_class["ACTCATEGORY_Predicted_Text"]==j
            elif value == "SAFEUNSAFE":
                Y1 = all_class["SAFE_UNSAFE_Actual_Text"]==i
                Y2 = all_class["SAFE_UNSAFE_Predicted_Text"]==j
            elif value == "INJPOTCAT":
                Y1 = all_class["INJPOT_Actual_Text"]==i
                Y2 = all_class["INJPOT_Predicted_Text"]==j
            Y = Y1 & Y2
            subset_matrix = all_class[Y]
            row_mat.append(subset_matrix.shape[0])
            if i==j:
                row_diag_mat.append(1)
            else: 
                row_diag_mat.append(0)
                non_diag_records+=subset_matrix.shape[0]
                
        confusion_matrix.append(row_mat)
        diag_valid_mat.append(row_diag_mat)
        
    for i in range(0,len(diag_valid_mat)):
        row_mat = diag_valid_mat[i]
        for j in range(0,len(row_mat)):
            ele = row_mat[j]
            if ele==0:
                diag_valid_mat[i][j] = confusion_matrix[i][j]/non_diag_records
    
            
  
    
    

    print(confusion_matrix)
    return {'data':[go.Heatmap(
                    z=diag_valid_mat, x = predicted_label , y = actual_label,text =  confusion_matrix, colorscale="greens",hoverinfo = "x+y+text")],
            'layout': go.Layout(xaxis = {"title":"Predicted"},yaxis = {"title":"Actual"})}

@dash_app.callback(
    dash.dependencies.Output('confusion_matrix_observation', 'children'),
    [dash.dependencies.Input('Confusion Matrix', 'clickData')],
    [dash.dependencies.State('my-dropdown','value')])

def confusion_matrix_observation(clickData,value):
    print(clickData)
    print(value)
    X = clickData
    Y = X["points"][0]
    print(Y)
    x_coordinate = Y['x']
    y_coordinate = Y['y']
    print(x_coordinate)
    all_class = pd.read_csv("All_Classifications.csv")
    
    if value=="ACTCAT":
        Y1 = all_class["ACTCATEGORY_Predicted_Text"]==x_coordinate
        Y2 = all_class["ACTCATEGORY_Actual_Text"]==y_coordinate
        Y = Y1&Y2
        subset_data = all_class[Y]
    elif value=="SAFEUNSAFE":
        Y1 = all_class["SAFE_UNSAFE_Predicted_Text"]==x_coordinate
        Y2 = all_class["SAFE_UNSAFE_Actual_Text"]==y_coordinate
        Y = Y1&Y2
        subset_data = all_class[Y]
    elif value=="INJPOTCAT":
        Y1 = all_class["INJPOT_Predicted_Text"]==x_coordinate
        Y2 = all_class["INJPOT_Actual_Text"]==y_coordinate
        Y = Y1&Y2
        subset_data = all_class[Y]
    
    return html.Div([
        dash_table.DataTable(
            data=subset_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in subset_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
            style_cell={'textAlign': 'left'}
        )
        
    ])

@dash_app.callback(
    dash.dependencies.Output('pyLDAvis_Output', 'children'),
    [dash.dependencies.Input('UC1_topic_modelling', 'n_clicks')])

def create_UC1_topic_modelling(n_clicks):
    if n_clicks>0:
        p_df = pd.read_csv('Usecase1_Upload.csv')
        p_df['DESCRIPTION'].dropna(inplace=True)
        docs = np.array(p_df['DESCRIPTION'])

        def docs_preprocessor(docs):
            tokenizer = RegexpTokenizer(r'\w+')
            for idx in range(len(docs)):
                #docs[idx] = [[word.lower() for word in docs[idx].split()] for line in data] # Convert to lowercase.
                docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

            # Remove numbers, but not words that contain numbers.
            docs = [[token for token in doc if not token.isdigit()] for doc in docs]

            # Remove words that are only one character.
            docs = [[token for token in doc if len(token) > 3] for doc in docs]

            # Lemmatize all words in documents.
            lemmatizer = WordNetLemmatizer()
            docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

            return docs

        docs = docs_preprocessor(docs)

        print(docs) 

        # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
        bigram = Phrases(docs, min_count=10)
        trigram = Phrases(bigram[docs])

        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
            for token in trigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)
        print('Number of unique words in initital documents:', len(dictionary))

        # Filter out words that occur less than 10 documents, or more than 20% of the documents.
        dictionary.filter_extremes(no_below=10, no_above=0.2)
        print('Number of unique words after removing rare and common words:', len(dictionary))

        corpus = [dictionary.doc2bow(doc) for doc in docs]
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))

        # Set training parameters.
        num_topics = 4
        chunksize = 500 # size of the doc looked at every pass
        passes = 20 # number of passes through documents
        iterations = 400
        eval_every = 1  # Don't evaluate model perplexity, takes too much time.

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                               alpha='auto', eta='auto', \
                               iterations=iterations, num_topics=num_topics, \
                               passes=passes, eval_every=eval_every)
        
        LDAvis_data_filepath = os.path.join('assets/sample_ldavis_uc1'+str(num_topics))
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if 1 == 1:
            LDAvis_prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)

        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath,'rb') as f:
            LDAvis_prepared = pickle.load(f)
        pyLDAvis.save_html(LDAvis_prepared, 'assets/sample_ldavis_uc1'+ str(num_topics) +'.html')
        
        Y = model.print_topics(num_words = 10)
        topic_words_list = []
        for i in range(0,len(Y)):
            topic_words_list.append(html.Br())
            topic_words_list.append("Topic "+str(i)+": ")
            X = Y[i][1]
            a = X.split("+")
            words_str = ""
            for j in range(0,len(a)):
                p = a[j].split("*")[1].replace("\"","")
                words_str += p
            topic_words_list.append(words_str)

        topic_words_list
        
        sent_topic_mapping = pd.DataFrame()
        sent_topic_mapping["Sentence"] = p_df['DESCRIPTION']
        sent_topic_mapping["Topic"] = "9"

        for i in range(0,len(corpus)):
            #sent_topic_mapping["Sentence"].iloc[i] = p_df['DESCRIPTION'].iloc[i]
            print(i)
            topic_dist = model[corpus[i]]
            inter_topic = []
            topic_scores = []
            for j in range(0,len(topic_dist)):
                inter_topic.append(topic_dist[j][0])
                topic_scores.append(topic_dist[j][1])
            print(inter_topic)
            print(topic_scores)
            sent_topic_mapping["Topic"].iloc[i]= inter_topic[topic_scores.index(max(topic_scores))]
        
        sent_topic_mapping["ID"] = sent_topic_mapping.index
        sent_topic_mapping.to_csv("Sentence_Topic_Distribution_UC1.csv")
        
        
        return html.Div([
            html.Div(topic_words_list), 
            html.Br(),
            html.Br(),
            html.Iframe(src = dash_app.get_asset_url("sample_ldavis_uc1"+str(num_topics)+ ".html"),width = 1200, height = 1000)
        ])
@dash_app.callback(
    dash.dependencies.Output('UC1_Topic_Distribution_Output', 'style'),
    [dash.dependencies.Input('UC1_Topic_Distribution', 'n_clicks')])

def UC1_show_topic_distribution(n_clicks):
    if n_clicks>0:
        return {'display':'block'}
    else: 
        return {'display':'none'}
    
    
@dash_app.callback(
    dash.dependencies.Output('UC1_Topic_Distribution_Output', 'figure'),
    [dash.dependencies.Input('UC1_Topic_Distribution', 'n_clicks')])

def UC1_create_topic_distribution(n_clicks):    
    if n_clicks>0:
        sent_topic_mapping = pd.read_csv('Sentence_Topic_Distribution_UC1.csv')
        ylabel = sent_topic_mapping.groupby("Topic")["ID"].nunique()
        xlabel = sent_topic_mapping["Topic"].unique()
        
        return {'data':[go.Bar(
                    x=xlabel,y=ylabel)],
            'layout': {
                'clickmode': 'event+select',

            }
        }

@dash_app.callback(
    dash.dependencies.Output('UC1_Observations', 'children'),
    [dash.dependencies.Input('UC1_Topic_Distribution_Output', 'clickData')])

def UC1_topic_observations(clickData):    
    
        
        
        X = clickData
        Y = X["points"][0]
        print(Y)
        x_coordinate = Y['x']
        #y_coordinate = Y['y']
        print(x_coordinate)
        sent_topic_mapping = pd.read_csv('Sentence_Topic_Distribution_UC1.csv')
        subset_data = sent_topic_mapping[sent_topic_mapping["Topic"]==int(x_coordinate)]
        
        
        return  html.Div([
            dash_table.DataTable(
            data=subset_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in subset_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
                style_cell={'textAlign': 'left'}
        )
        ])

page_2_layout = html.Div([
    html.Div(children = [
     #       dcc.Link('Safety Observations Classification and Theme identification', href='/Observations'),
         html.H1('Incident Injury Keyword extraction and Theme identification', style={'background-image': 'url(https://erm.com/)','color': 'green','size':'30'}),
        html.Img(src = dash_app.get_asset_url(image_filename2)),    
        #html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode())),
          html.Br(),
            dcc.Link('Main Page', href='/'),
            html.Br(),
            html.Br(),
            dcc.Upload(id='upload_data_tab2',
                        children=html.Div(['Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            }), 
            html.Div(id='output-data-upload_tab2'), 
            html.Hr(),
            html.Br(),
            html.Button("Process Data",id="process_uc2"),
            html.Div(id="NER Success"),
            html.Br(),
            html.Br(),
            html.Button("Visualization",id = "uc2_NER_vis"),
            dcc.Graph(id="Injury NER",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
            html.Div(id="Injury_NER_Observation"),
            dcc.Graph(id="Body Part NER",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
            html.Div(id="Body_Part_Observation"),
            html.Button('Time Trend Analysis', id='Time_Trend_button'),
            dcc.Graph(id="Injury_Trend",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
            html.Div(id="Injury_Trend_Observations"),
            dcc.Graph(id="Body_Part_Trend",figure={'layout': {
                                    'clickmode': 'event+select'}},style = {'display':'none'}),
            html.Div(id="Body_Part_Trend_Observations"),
            html.Button('Topic Modelling', id='UC2_topic_modelling'),
            html.Div(id="pyLDAvis_Output_UC2"),
            html.Br(),
            html.Br(),
            html.Button('Topic Distribution',id="UC2_Topic_Distribution"),
            dcc.Graph(id = "UC2_Topic_Distribution_Output",figure= {'layout': {
                                            'clickmode': 'event+select'}},style = {'display':'none'}),
            html.Div(id= "UC2_Observations")
            
        
    ])
])


def parse_contents_uc2(contents,filename):
    #print(contents)
    #print(filename)
    print("parse contents code Usecase 2")
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            raw_data = pd.read_csv(
                io.StringIO(decoded.decode('ISO-8859-1')))
            #Step 1: Adding safe and unsafe column
            preprocess_data = raw_data
            preprocess_data["NO_STOPWORDS_FINAL_NARRATIVE"] = remove_stopwords(preprocess_data["Final Narrative"])
    
            preprocess_data.to_csv("Usecase2_Upload.csv")
            #no_sw_data = create_text_data(preprocess_data.NO_STOPWORDS_DESCRIPTION_CACHED)
                      
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            raw_data = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        #html.Div(children=draw_wordcloud(no_sw_data,"sample_data")),
        dash_table.DataTable(
            data=raw_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in raw_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
            style_cell={'textAlign': 'left'}
        )
    ])




@dash_app.callback(Output('output-data-upload_tab2','children'),
              [Input('upload_data_tab2','contents')],
              [State('upload_data_tab2','filename')])

def Update_output_tab2(list_of_contents,list_of_names):
    if list_of_contents is not None: 
        children = [
            parse_contents_uc2(list_of_contents, list_of_names) ]
        return children

@dash_app.callback(Output('NER Success','children'),
             [Input('process_uc2','n_clicks')])

def NER_Injury_Body_Part(n_clicks):
    if n_clicks>0:
        
        data = pd.read_csv("Usecase2_Upload.csv")
        
        injury_list = list(data["Nature of Injury"].unique())
        X= str(injury_list)
        X = X.replace("unspecified","")
        X = X.replace("heat (thermal) ","")
        X = X.replace("Heat (thermal) ","")
        X = X.replace(" n.e.c. ","")
        X = X.replace(" n.e.c.","")
        X = X.replace("[","")
        X = X.replace("]","")
        X = X.replace("'","")
        X = X.split(",")
        injury_data = X
        injury_data
        
        body_part_list = list(data["Affected Body Part"].unique())
        Y= str(body_part_list)
        Y = Y.replace("(s)","s")
        Y = Y.replace(", n.e.c.","")
        Y = Y.replace(",unspecified","")
        Y = Y.replace("unspecified","")
        Y = Y.replace("Nonclassifiable","")
        Y = Y.replace("[","")
        Y = Y.replace("]","")
        Y = Y.replace("'","")
        Y = Y.split(",")
        body_part_data = Y
        body_part_data
        
        stemmer = SnowballStemmer('english')
        def string_matcher(dict_str,sent):
            result = 0
            #print dict_str
            #print sent


            if dict_str in sent:
                dict_str_split = dict_str.split(" ")
                dict_str_len = len(dict_str_split)
                sent_split = sent.split(" ")
                #print sent_split
                words_sent = len(sent_split)+1
                for i in range(0,words_sent-dict_str_len):
                    ctr = 0
                    #print i
                    for j in range(i,dict_str_len+i):
                        clean_sent = sent_split[j].replace(",","")
                        clean_sent = clean_sent.lower()
                        dict_str_split[ctr] = dict_str_split[ctr].lower()
                        if stemmer.stem(dict_str_split[ctr]) == stemmer.stem(clean_sent):
                            ctr = ctr+1
                        else: 
                            break

                    #print ctr
                    if ctr==dict_str_len:
                        result = 1
            return result
        
        data["Injury_predictions"] = "1"
        data["Body_Part_Predictions"] = "2"
        for i in range(0,data.shape[0]):
            print(i)
            text = data["Final Narrative"].iloc[i]
            tokenized_words = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            stemmed_tokens_no_sw = [stemmer.stem(word) for word in tokenized_words]
            stemmed_text = " ".join([j for j in stemmed_tokens_no_sw])
            injury_str = ""
            for j in injury_data:
                #print(j)
                j = j.strip()
                if len(j) !=0:
                    tokenized_words_j = [word for sent in nltk.sent_tokenize(j) for word in nltk.word_tokenize(sent)]
                    stem_j = [stemmer.stem(word) for word in tokenized_words_j]
                    j_text = " ".join([j for j in stem_j])
                    valid = string_matcher(j_text.lower(),stemmed_text.lower())
                    if valid ==1:
                        injury_str += j + "|"
            data["Injury_predictions"].iloc[i] = injury_str
            body_part_str = ""
            for k in body_part_data:
                #print(j)
                k = k.strip()
                if len(k)!=0:
                    tokenized_words_k = [word for sent in nltk.sent_tokenize(k) for word in nltk.word_tokenize(sent)]
                    stem_k = [stemmer.stem(word) for word in tokenized_words_k]
                    p_text = " ".join([p for p in stem_k])
                    valid = string_matcher(p_text.lower(),stemmed_text.lower())
                    if valid ==1:
                        body_part_str += k + "|"
            data["Body_Part_Predictions"].iloc[i] = body_part_str
        data["New_ID"] = data.index
        data.to_csv("Injury_Body_Ner.csv")
        ind = 1
        
        if ind==1:
            return html.Div([html.H5("NER Success")])

@dash_app.callback(
    dash.dependencies.Output('Injury NER', 'style'),
    [dash.dependencies.Input('uc2_NER_vis', 'n_clicks')])

def show_Injury_graph(n_clicks):
    if n_clicks>0:
        return {'display':'block'}
    else: 
        return {'display':'none'}
    
@dash_app.callback(
    dash.dependencies.Output('Injury NER', 'figure'),
    [dash.dependencies.Input('uc2_NER_vis', 'n_clicks')])

def create_Injury_graph(n_clicks):
    if n_clicks>0:
        NER_data = pd.read_csv("Injury_Body_Ner.csv")
        #xvalue = unique_injury = NER_data["Injury_predictions"].unique()
        yvalue = NER_data.groupby("Injury_predictions")['New_ID'].nunique()
        xvalue = yvalue.index
        return {'data':[go.Bar(
                    x=xvalue,y=yvalue)],
            'layout': {
                'clickmode': 'event+select',

            }}

@dash_app.callback(
    dash.dependencies.Output('Injury_NER_Observation', 'children'),
    [dash.dependencies.Input('Injury NER', 'clickData')])

def Injury_NER_observations(clickData): 
        X = clickData
        Y = X["points"][0]
        print(Y)
        x_coordinate = Y['x']
        #y_coordinate = Y['y']
        print(x_coordinate)
        injury_body_ner_data = pd.read_csv('Injury_Body_Ner.csv')
        subset_data = injury_body_ner_data[injury_body_ner_data["Injury_predictions"]==x_coordinate]
        
        
        return  html.Div([
            dash_table.DataTable(
            data=subset_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in subset_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
                style_cell={'textAlign': 'left'}
        )
        ])

@dash_app.callback(
    dash.dependencies.Output('Body Part NER', 'style'),
    [dash.dependencies.Input('uc2_NER_vis', 'n_clicks')])

def show_bdy_part_graph(n_clicks):
    if n_clicks>0:
        return {'display':'block'}
    else: 
        return {'display':'none'}
    
@dash_app.callback(
    dash.dependencies.Output('Body Part NER', 'figure'),
    [dash.dependencies.Input('uc2_NER_vis', 'n_clicks')])

def create_bdy_part_graph(n_clicks):
    if n_clicks>0:
        NER_data = pd.read_csv("Injury_Body_Ner.csv")
        #xvalue  = NER_data["Body_Part_Predictions"].unique()
        yvalue = NER_data.groupby("Body_Part_Predictions")['New_ID'].nunique()
        xvalue = yvalue.index
        return {'data':[go.Bar(
                    x=xvalue,y=yvalue)],
            'layout': {
                'clickmode': 'event+select',

            }}       

@dash_app.callback(
    dash.dependencies.Output('Body_Part_Observation', 'children'),
    [dash.dependencies.Input('Body Part NER', 'clickData')])

def Body_Part_NER_observations(clickData): 
        X = clickData
        Y = X["points"][0]
        print(Y)
        x_coordinate = Y['x']
        #y_coordinate = Y['y']
        print(x_coordinate)
        injury_body_ner_data = pd.read_csv('Injury_Body_Ner.csv')
        subset_data = injury_body_ner_data[injury_body_ner_data["Body_Part_Predictions"]==x_coordinate]
        
        
        return  html.Div([
            dash_table.DataTable(
            data=subset_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in subset_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
                style_cell={'textAlign': 'left'}
        )
        ])

@dash_app.callback(
    dash.dependencies.Output('Body_Part_Trend', 'style'),
    [dash.dependencies.Input('Time_Trend_button', 'n_clicks')])

def show_body_part_trend(n_clicks):
    if n_clicks>0:
        return {'display':'block'}
    else: 
        return {'display':'none'}
    
@dash_app.callback(
    dash.dependencies.Output('Body_Part_Trend', 'figure'),
    [dash.dependencies.Input('Time_Trend_button', 'n_clicks')])

def create_body_part_trend(n_clicks):
    if n_clicks>0:
        injury_data = pd.read_csv("Injury_Body_Ner.csv")
        EventDate = pd.to_datetime(injury_data["EventDate"])
        date_month = []
        for i in EventDate:
            date_month.append(i.month)
        injury_data["Month"] = date_month
        
        monthly_dict = {"1":"January",
                "2":"February",
                "3":"March",
                "4":"April",
                "5":"May",
                "6":"June",
                "7":"July",
                "8":"August",
                "9":"September",
                "10":"October",
                "11":"November",
                "12":"December"}
        unique_body_part = injury_data["Body_Part_Predictions"].unique()


        data = []
        for i in unique_body_part: 
            unique_body_part_data = injury_data[injury_data["Body_Part_Predictions"]==i]
            ylabel = unique_body_part_data.groupby("Month")["New_ID"].nunique()
            xlabel = ylabel.index
            xlabel_text = [monthly_dict.get(str(j)) for j in xlabel]
            #month_i = monthly_dict.get(str(i))
            data.append(go.Bar(name=i,x = xlabel_text,y=ylabel))


        
        return {'data':data,
            'layout': {
                'clickmode': 'event+select',
                

            }}

@dash_app.callback(
    dash.dependencies.Output('Injury_Trend', 'style'),
    [dash.dependencies.Input('Time_Trend_button', 'n_clicks')])

def show_injury_trend(n_clicks):
    if n_clicks>0:
        return {'display':'block'}
    else: 
        return {'display':'none'}
    
@dash_app.callback(
    dash.dependencies.Output('Injury_Trend', 'figure'),
    [dash.dependencies.Input('Time_Trend_button', 'n_clicks')])

def create_injury_trend(n_clicks):
    if n_clicks>0:
        injury_data = pd.read_csv("Injury_Body_Ner.csv")
        EventDate = pd.to_datetime(injury_data["EventDate"])
        date_month = []
        for i in EventDate:
            date_month.append(i.month)
        injury_data["Month"] = date_month
        
        monthly_dict = {"1":"January",
                "2":"February",
                "3":"March",
                "4":"April",
                "5":"May",
                "6":"June",
                "7":"July",
                "8":"August",
                "9":"September",
                "10":"October",
                "11":"November",
                "12":"December"}
        unique_injury = injury_data["Injury_predictions"].unique()


        data = []
        for i in unique_injury: 
            unique_injury_data = injury_data[injury_data["Injury_predictions"]==i]
            ylabel = unique_injury_data.groupby("Month")["New_ID"].nunique()
            xlabel = ylabel.index
            xlabel_text = [monthly_dict.get(str(j)) for j in xlabel]
            #month_i = monthly_dict.get(str(i))
            data.append(go.Bar(name=i,x = xlabel_text,y=ylabel))


        
        return {'data':data,
            'layout': {
                'clickmode': 'event+select',
                

            }}  

@dash_app.callback(
    dash.dependencies.Output('pyLDAvis_Output_UC2', 'children'),
    [dash.dependencies.Input('UC2_topic_modelling', 'n_clicks')])

def Topic_Modelling_UC2(n_clicks):
    if n_clicks>0:
        p_df = pd.read_csv('Usecase2_Upload.csv')
        p_df['Final Narrative'].dropna(inplace=True)
        docs = np.array(p_df['Final Narrative'])

        

        def docs_preprocessor(docs):
            tokenizer = RegexpTokenizer(r'\w+')
            for idx in range(len(docs)):
                #docs[idx] = [[word.lower() for word in docs[idx].split()] for line in data] # Convert to lowercase.
                docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

            # Remove numbers, but not words that contain numbers.
            docs = [[token for token in doc if not token.isdigit()] for doc in docs]

            # Remove words that are only one character.
            docs = [[token for token in doc if len(token) > 3] for doc in docs]

            # Lemmatize all words in documents.
            lemmatizer = WordNetLemmatizer()
            docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

            return docs

        docs = docs_preprocessor(docs)

        #print(docs) 

        
        # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
        bigram = Phrases(docs, min_count=10)
        trigram = Phrases(bigram[docs])

        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
            for token in trigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)


        

        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)
        #print('Number of unique words in initital documents:', len(dictionary))

        # Filter out words that occur less than 10 documents, or more than 20% of the documents.
        dictionary.filter_extremes(no_below=10, no_above=0.2)
        #print('Number of unique words after removing rare and common words:', len(dictionary))

        corpus = [dictionary.doc2bow(doc) for doc in docs]
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))

        

        # Set training parameters.
        num_topics = 4
        chunksize = 500 # size of the doc looked at every pass
        passes = 20 # number of passes through documents
        iterations = 400
        eval_every = 1  # Don't evaluate model perplexity, takes too much time.

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                               alpha='auto', eta='auto', \
                               iterations=iterations, num_topics=num_topics, \
                               passes=passes, eval_every=eval_every)
        
        
        
        
        LDAvis_data_filepath = os.path.join('assets/sample_ldavis_uc2'+str(num_topics))
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if 1 == 1:
            LDAvis_prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)

        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath,'rb') as f:
            LDAvis_prepared = pickle.load(f)
        pyLDAvis.save_html(LDAvis_prepared, 'assets/sample_ldavis_uc2'+ str(num_topics) +'.html')
        
        Y = model.print_topics(num_words = 10)
        topic_words_list = []
        for i in range(0,len(Y)):
            topic_words_list.append(html.Br())
            topic_words_list.append("Topic "+str(i)+": ")
            X = Y[i][1]
            a = X.split("+")
            words_str = ""
            for j in range(0,len(a)):
                p = a[j].split("*")[1].replace("\"","")
                words_str += p
            topic_words_list.append(words_str)

        topic_words_list
        
        
        sent_topic_mapping = pd.DataFrame()
        sent_topic_mapping["Sentence"] = p_df['Final Narrative']
        sent_topic_mapping["Topic"] = "9"

        for i in range(0,len(corpus)):
            #sent_topic_mapping["Sentence"].iloc[i] = p_df['DESCRIPTION'].iloc[i]
            #print(i)
            topic_dist = model[corpus[i]]
            inter_topic = []
            topic_scores = []
            for j in range(0,len(topic_dist)):
                inter_topic.append(topic_dist[j][0])
                topic_scores.append(topic_dist[j][1])
            #print(inter_topic)
            #print(topic_scores)
            sent_topic_mapping["Topic"].iloc[i]= inter_topic[topic_scores.index(max(topic_scores))]
        
        sent_topic_mapping["ID"] = sent_topic_mapping.index
        sent_topic_mapping.to_csv("Sentence_Topic_Distribution_UC2.csv")
        
        return html.Div([
    html.Div(topic_words_list),
    html.Br(),
    html.Br(),
    html.Iframe(src = dash_app.get_asset_url("sample_ldavis_uc2"+str(num_topics)+ ".html"),width = 1200, height = 1000)
    
        
        ])
@dash_app.callback(
    dash.dependencies.Output('UC2_Topic_Distribution_Output', 'style'),
    [dash.dependencies.Input('UC2_Topic_Distribution', 'n_clicks')])

def UC2_show_topic_distribution(n_clicks):
    if n_clicks>0:
        return {'display':'block'}
    else: 
        return {'display':'none'}
    
    
@dash_app.callback(
    dash.dependencies.Output('UC2_Topic_Distribution_Output', 'figure'),
    [dash.dependencies.Input('UC2_Topic_Distribution', 'n_clicks')])

def UC2_create_topic_distribution(n_clicks):    
    if n_clicks>0:
        sent_topic_mapping = pd.read_csv('Sentence_Topic_Distribution_UC2.csv')
        ylabel = sent_topic_mapping.groupby("Topic")["ID"].nunique()
        xlabel = sent_topic_mapping["Topic"].unique()
        
        return {'data':[go.Bar(
                    x=xlabel,y=ylabel)],
            'layout': {
                'clickmode': 'event+select',

            }
        }

@dash_app.callback(
    dash.dependencies.Output('UC2_Observations', 'children'),
    [dash.dependencies.Input('UC2_Topic_Distribution_Output', 'clickData')])

def UC2_topic_observations(clickData):    
    
        
        
        X = clickData
        Y = X["points"][0]
        print(Y)
        x_coordinate = Y['x']
        #y_coordinate = Y['y']
        print(x_coordinate)
        sent_topic_mapping = pd.read_csv('Sentence_Topic_Distribution_UC2.csv')
        subset_data = sent_topic_mapping[sent_topic_mapping["Topic"]==int(x_coordinate)]
        
        
        return  html.Div([
            dash_table.DataTable(
            data=subset_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in subset_data.columns],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'overflowX': 'scroll'
                    },
                style_cell={'textAlign': 'left'}
        )
        ])

    
    
    
    

# Update the index
@dash_app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/Observations':
        return page_1_layout
    elif pathname == '/Incidents':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

    
if __name__ == '__main__':
    dash_app.run_server()
