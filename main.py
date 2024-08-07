import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit_authenticator import Authenticate
import yaml
import openai
from yaml.loader import SafeLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

st.set_page_config(
    page_title="Buscador de Propuestas",
    layout="wide",
    menu_items={
        'Get help': 'https://www.notus.cl/', "Report a bug": 'https://www.notus.cl/',"About": 'https://www.notus.cl/'}
)

if "visibility" not in st.session_state:
    st.session_state.disabled = False

openai.api_key=st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_model():
    model = ChatOpenAI(model_name="gpt-4",temperature=0)
    return model

llm = load_model()

prompt_template = """Eres experto encontrando semejanzas entre proyectos.

A continuación, recibirás la descripción de proyectos.

{context}

Se te va a pedir que busques proyectos según ciertos criterios que se te especificarán.

Estos criterios pueden ser: tipo de servicio, metodología, tipo de problema, industria.

Tú tienes que encontrar los proyectos que más se ajusten a lo que te pide. 

Siempre tienes que responder con los proyectos que mejor se ajusten a los criterios y justificar tu respuesta.

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# DATAFRAME

df=pd.read_excel('BBDD_Proyectos.xlsx').rename(columns={'Nombre Proyecto':'Proyecto'})
df['Cliente']=df.ID.map(lambda x: st.secrets[str(x)])
df=df[['Proyecto',
      'Cliente',
      'Tipo de Servicio',
      'Metodología',
      'Industria',
      'Tipo de Problema',
      'Descripción']]




with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(location='main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Bienvenido *{name}*')
# Título
    st.title(':paperclip: Buscador de Propuestas')

# Descripción
    st.info('Esta herramienta te ayuda a buscar propuestas pasadas según los criterios que selecciones.')
# st.subheader('Carga propuestas en formato PDF:')
# doc=st.file_uploader('Sube tu(s) documento(s) aquí:',help='Sube documento(s) en formato pdf',type='pdf',accept_multiple_files=True)

    st.header('Utiliza los siguiente filtros:')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        serv=df['Tipo de Servicio'].unique()
        serv.sort()
        options_serv = st.multiselect(label='Selecciona Tipo(s) de Servicio(s)',
                         options=serv,
                         help='Selecciona los tipos de servicio que deseas buscar',
                         placeholder='Selecciona Tipo(s) de Servicio(s)')

    with col2: 
        met=df['Metodología'].unique()
        met.sort()
        options_met = st.multiselect(label='Selecciona Metodología(s)',
                         options=met,
                         help='Selecciona las metodologías que deseas buscar',
                         placeholder='Selecciona Metodología(s)')

    with col3: 
        ind=df['Industria'].unique()
        ind.sort()
        options_ind = st.multiselect(label='Selecciona Industria(s)',
                         options=ind,
                         help='Selecciona las industrias que deseas buscar',
                         placeholder='Selecciona Industria(s)')
    
    with col4: 
        prob=df['Tipo de Problema'].unique()
        prob.sort()
        options_prob = st.multiselect(label='Selecciona Tipo(s) de Problema(s)',
                         options=prob,
                         help='Selecciona los tipos de problemas que deseas buscar',
                         placeholder='Selecciona Tipo(s) de Problema(S)')



    if options_serv == [] and options_met == [] and options_ind == [] and options_prob == []:
        st.dataframe(df.drop_duplicates(subset=['ID'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True), hide_index=True)

    elif options_serv != [] and options_met == [] and options_ind == [] and options_prob == []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)].drop_duplicates(subset=['ID', 'Tipo de Servicio'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met != [] and options_ind == [] and options_prob == []:
        df_aux=df[df['Metodología'].isin(options_met)].drop_duplicates(subset=['ID', 'Metodología'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met == [] and options_ind != [] and options_prob == []:
        df_aux=df[df['Industria'].isin(options_ind)].drop_duplicates(subset=['ID', 'Industria'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met != [] and options_ind == [] and options_prob == []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Metodología'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met == [] and options_ind != [] and options_prob == []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Industria.isin(options_ind)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Industria'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met != [] and options_ind != [] and options_prob == []:
        df_aux=df[df['Metodología'].isin(options_met)][df[df['Metodología'].isin(options_met)].Industria.isin(options_ind)].drop_duplicates(subset=['ID', 'Metodología', 'Industria'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met != [] and options_ind != [] and options_prob == []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)][df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)].Industria.isin(options_ind)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Metodología','Industria'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met == [] and options_ind == [] and options_prob != []:
        df_aux=df[df['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met == [] and options_ind == [] and options_prob != []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met != [] and options_ind == [] and options_prob != []:
        df_aux=df[df['Metodología'].isin(options_met)][df[df['Metodología'].isin(options_met)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Metodología', 'Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met == [] and options_ind != [] and options_prob != []:
        df_aux=df[df['Industria'].isin(options_ind)][df[df['Industria'].isin(options_ind)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Industria', 'Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met != [] and options_ind == [] and options_prob != []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)][df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Metodología','Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met == [] and options_ind != [] and options_prob != []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Industria.isin(options_ind)][df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Industria.isin(options_ind)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Industria','Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv == [] and options_met != [] and options_ind != [] and options_prob != []:
        df_aux=df[df['Metodología'].isin(options_met)][df[df['Metodología'].isin(options_met)].Industria.isin(options_ind)][df[df['Metodología'].isin(options_met)][df[df['Metodología'].isin(options_met)].Industria.isin(options_ind)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Metodología', 'Industria','Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)

    elif options_serv != [] and options_met != [] and options_ind != [] and options_prob != []:
        df_aux=df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)][df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)].Industria.isin(options_ind)][df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)][df[df['Tipo de Servicio'].isin(options_serv)][df[df['Tipo de Servicio'].isin(options_serv)].Metodología.isin(options_met)].Industria.isin(options_ind)]['Tipo de Problema'].isin(options_prob)].drop_duplicates(subset=['ID', 'Tipo de Servicio', 'Metodología','Industria','Tipo de Problema'], ignore_index=True).drop(columns=['ID']).reset_index(drop=True).copy()
        n=df_aux.Proyecto.nunique()
        if n==0:
            st.warning('No se encontraron proyectos')
        else:
            st.success('Se encontraron '+str(n)+' proyectos únicos')
            st.dataframe(df_aux, hide_index=True)



    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    loader = PyPDFLoader("Notus Proyectos.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150)
    splits = text_splitter.split_documents(pages)
    docsearch =FAISS.from_documents(splits, embedding)     
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    st.subheader('¿No encontraste un proyecto? ¡Te ayudo a buscar uno similar!')
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": 'Hola 👋 Intenta algo así: "Dime los proyectos más similares a..." :)'})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if pregunta := st.chat_input("Escribe aquí lo que necesitas encontrar"):
        with st.chat_message('user'):
            st.markdown(pregunta)
        st.session_state.messages.append({'role':'user',"content": pregunta})
        respuesta=qa.run(pregunta)
        message = st.chat_message("assistant")
        message.empty()
        message.write(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
elif authentication_status == False:
    st.error('Usuario/Contraseña incorrectos')
elif authentication_status == None:
    st.warning('Por favor, ingresa tu nombre de usuario y contraseña')
