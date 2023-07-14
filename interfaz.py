import streamlit as st
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline
import os
os.environ['CURL_CA_BUNDLE'] = ""

#############################################
#############################################
############################################
def complete_text(reader, num_page):
    page_0 = reader.pages[0]
    text  = page_0.extract_text()
    for i in range(1, num_page):
        page = reader.pages[i]
        text = text + " " + page.extract_text()
    return(text)

#############################################
#############################################
############################################
def patron_deleting(text, patron_list):
    for my_patron in patron_list:
        text = text.replace(my_patron, " ")
    return text

#############################################
#############################################
############################################

def simi(context_v, question_v):
    a = question_v
    b = context_v
    return(cosine_similarity([a], [b])[0,0])

#############################################
#############################################
############################################
st.set_page_config(
    page_title="Home",
    page_icon="🚀",
)

st.write("# Cuentame, Yo te respondo! 👋")

st.write(" Necesito un documento")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    number_of_pages = len(reader.pages)
else:
    mypdf = open('1.pdf', mode='rb')
    reader = PyPDF2.PdfReader(mypdf)
    number_of_pages = len(reader.pages)

    
st.write("Desea eliminar patrones del texto")
my_n = st.text_input('Escribe aqui si o no', ) 
patron_list = []
if my_n == "si":
    n = 1
    while my_n == 'si':
        st.write("Agrega el patron renglon por renglon")
        user_input = st.text_input('Escribe aqui el patron numero ' + str(n), ) 
        patron_list.append(user_input)
        n = n + 1
        st.write("Deasea Agregar patron no " + str(n) + "?")
        my_n = st.text_input('Escribe aqui si o no por ' + str(n) + " vez", )


my_text = complete_text(reader, number_of_pages)
my_text_clean = patron_deleting(my_text, patron_list)
my_text_clean_lower = my_text_clean.lower()

paragrphs = my_text_clean_lower.split(".  ")
#st.write(paragrphs)

model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
vectors = model.encode(paragrphs).tolist()

original_context_df = {'paragrphs': paragrphs,
                       'vectors': vectors}

original_context_df = pd.DataFrame(original_context_df)
#st.write(original_context_df)
###############################################
###############################################
###############################################
###############################################
st.write("Has tu pregunta")


question = st.text_input("No olvides el signo ? ", 'Qué hora es?')
if question is  None:
    question = 'Qué hora es?'

print(question)
    
question_v = model.encode(question).tolist()


original_context_df['cos sim'] = original_context_df['vectors'].map(lambda x: cosine_similarity([x], [question_v])[0,0] )
original_context_df['corr'] = original_context_df['vectors'].map(lambda x: pearsonr(x, question_v)[0])
#st.write(original_context_df)

context_df_ordered_s = original_context_df.sort_values(by=['cos sim'], ascending= False).head(5).sort_index().reset_index(drop= True)

best_context = " "
for i in range(0, 5):
    best_context = best_context + '\n' + context_df_ordered_s['paragrphs'][i]   
#st.write(best_context)

model2 = BertForQuestionAnswering.from_pretrained("Josue/BETO-espanhol-Squad2")
tokenizer = AutoTokenizer.from_pretrained("Josue/BETO-espanhol-Squad2")

nlp = pipeline("question-answering", model = model2, tokenizer = tokenizer)

qa = nlp({
    #'question': question_df["question"][0],
    'question': question,
    'context': best_context
    })

st.write("Espero te ayude mi respuesta 😉")
st.write(qa["answer"])


