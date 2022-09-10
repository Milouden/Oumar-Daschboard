# TO RUN: $streamlit run dashboard-streamlit.py

import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import shap
import time
from custtransformer import CustTransformer
from dashboard_functions import plot_boxplot_var_by_target
from dashboard_functions import plot_scatter_projection




# Configuration de la page streamlit
st.set_page_config(page_title='Tableau de bord de notation des clients demandeurs de prêts :',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')
#################################
#################################
#################################
# Affichage du titre
st.title('Tableau de bord de notation des demandeurs de prêts :')
st.header("Oumar Gueye - Data Scientist - OC - project 7")
path = "logo2.png"
image = Image.open(path)
st.sidebar.image(image, width=300)
#st.text("Implémentez un modèle de scoring")
st.subheader('Implémentez un modèle de scoring')
#st.text("les informations relatives à un client ou un groupe de clients :")
#id_input = st.text_input("Veuillez saisir l'identifiant d'un client:", )


@st.cache
def load_data (nrows):
    data = pd.read_parquet('app_test.parquet', nrows=nrows)
    data.drop(columns = {'Unnamed: 0'}  , inplace = True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(0, inplace=True)
    return data

data_load = st.text('Chargement des données...')
data = load_data() 
# 50
#st.write(data.head(5))



# Affichage du titre et des données relatives au client sélectionné ------------
st.sidebar.header('Sélection du numéro du client')
id_client = st.sidebar.selectbox("Veuillez sélectionner l'identifiant d'un client:", data['SK_ID_CURR'])

#st.subheader('Les données relatives au client sélectionné')

data_client = data.loc[data['SK_ID_CURR'] == int(id_client)]
#data_client.columns = ['Informations clients']
st.write(data_client)


check_box2 = st.sidebar.checkbox(label = 'Description du projet')
if (check_box2):
    path1 = "logo.png"
    image1 = Image.open(path1)
    st.image(image1, width=300) 
    #st.title('Implémentez un modèle de scoring')
    st.write('''
    **L'objectif du projet :**
    - Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique. 
    - Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client.
    **Statut de remboursement du prêt**
    - **la valeur 0** : signifie que le prêt est remboursé
    - **la valeur 1** : signifie que le prêt n'est pas remboursé.
    ''')

    
    
def main() :

    @st.cache
    def load_data():
        #PATH = 'dataset/'
        #données test après feature engeniering
        df = pd.read_csv('app_test.csv')
        
        #données test avant feature engeniering
        data_test = pd.read_csv('app_test.csv')
        
        #données train avant feature engeniering
        data_train = pd.read_parquet('app_train.cvs')
        
        #données test avant feature engeniering
        #X_test = pd.read_csv(PATH+'X_test.parquet',encoding="ISO-8859-1", on_bad_lines='skip',lineterminator='\n')
        
        #données train avant feature engeniering
        #y_test = pd.read_csv(PATH+'y_test.csv')        
        
        #description des features
        description = pd.read_csv('HomeCredit_columns_description.csv', 
                                      usecols=['Row', 'Description'], \
                                  index_col=0, encoding='unicode_escape')

        return df, data_test, data_train,  description   # ,X_test 
         
        
        

check_box1 = st.sidebar.checkbox(label = 'Echantillonnage des données')
if (check_box1):
    
    st.subheader("Les dimensions de notre échantilon de données")
    st.write(data.shape)
    
    st.subheader("Les 5 premières lignes de notre Dataframe")
    st.write(data.head(5))
    
    st.subheader('Les Statistiques de base de notre Dataframe')
    st.write(data.describe().T)
    
check_box4 = st.sidebar.checkbox(label = 'Décision de crédit')
if (check_box4):

    #########################################################################################################
    #Titre principal

    html_temp = """
    <div style="background-color:pink; padding:10px; border-radius:5px">
    <h1 style="color: white; text-align:center"> </h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center"></p>
    """
            
    st.markdown(html_temp, unsafe_allow_html=True)
    with st.sidebar:
        ## st.header(" Prêt à dépenser")
        
        st.write("#### Veuillez choisir l'identificateur du Client...")
        #st.write("**ID Client est  :**", id_client)
        id_list = data["SK_ID_CURR"].tolist()
        # id_client = st.number_input("Sélectionner l'identifiant du client" , step = 1 , value = 100001 )
        id_client = st.number_input(" " , step = 1 , value = 100001 )
        # id_client = st.selectbox(
        #    "Sélectionner l'identifiant du client", id_list)

####    with st.expander("🤔 A quoi sert cette application ?"):
####        st.write("Ce dashboard interactif à destination des gestionnaires de la relation client de l'entreprise **Prêt à dépenser** permet de comprendre et interpréter les décisions potentielles (prédictions faites par un modèle d'apprentissage) d'ottroi ou non de crédit aux clients") 
####        st.text('\n') 
####        st.write("**Objectif**:  répondre au soucis de transparence vis-à-vis des décisions d’octroi de crédit qui va tout à fait dans le sens des valeurs que l’entreprise veut incarner")
        ## st.image(LOGO_IMAGE)

def main() :
    @st.cache
    def load_model():
        '''loading the trained model'''
        return pickle.load(open('ModelClassifier.pkl', 'rb'))

    @st.cache
    def get_client_info(data, id_client):
        client_info = data[data['SK_ID_CURR']==int(id_client)]
        return client_info
    
###########################################################################################################################################
###########################################################################################################################################


    #Afficher l'ID Client sélectionné
    #st.write("**ID Client est  :**", id_client)
    id_list = data["SK_ID_CURR"].tolist()
    if (int(id_client) in id_list):
        st.markdown(" ✅ **Ce client est dans notre base de donnée...**")

        client_info = get_client_info(data, id_client)
         #-------------------------------------------------------
        # Afficher les informations du client
        #-------------------------------------------------------

        personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",
            
            'AMT_GOODS_PRICE'         : "Prix des biens",
          ##  'NAME_TYPE_SUITE'         :  "Accompagnateur du client",
          ##  'NAME_EDUCATION_TYPE'     :   "Niveau de scolarité le plus élevé du client",
            'NAME_HOUSING_TYPE'       :   "Situation de logement du client",
            'DAYS_REGISTRATION'       :   "Nbr jours avant la demande de modification de l'inscription",
            'OWN_CAR_AGE'             :   "Age de la voiture",
          ##  'FLAG_MOBIL'              :   "fourni un téléphone portable",
            'OCCUPATION_TYPE'         :   "Occupation",
            'CNT_FAM_MEMBERS'         :   "nombre de membre de la famille",
            'REGION_RATING_CLIENT'    :    "Notre évaluation de la région",
        }
        
        default_list=\
        ["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","NB ANNEES EMPLOI","NIVEAU EDUCATION","Prix des biens",
      ##   "Accompagnateur du client","Niveau de scolarité le plus élevé du client","Situation de logement du client",
       ##  "Nbr jours avant la demande de modification de l'inscription","Age de la voiture","fourni un téléphone portable",
         "Occupation","nombre de membre de la famille","Notre évaluation de la région",
        ]
        numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'DAYS_REGISTRATION' , 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','NAME_HOUSING_TYPE']

        rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
        horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]
        
        
    else:    
        st.markdown("**❌  Ce client n'existe pas dans notre base de donnée...**")
             


check_box3 = st.sidebar.checkbox(label = 'Visualisations des graphes')
if (check_box3):
    
    # LOGO_IMAGE = "logo.png"
    SHAP_GENERAL = "logo2.png"
    # shap_general = "feature_importance.png"

    with st.sidebar:
        ## st.header(" Prêt à dépenser")
        
        #st.write("## Identificateur du Client")
        #st.write("**ID Client est  :**", id_client)
        #id_list = data["SK_ID_CURR"].tolist()
        # id_client = st.number_input("Sélectionner l'identifiant du client" , step = 1 , value = 100001 )
        #id_client = st.number_input(" " , step = 1 , value = 100001 )
        # id_client = st.selectbox(
        #    "Sélectionner l'identifiant du client", id_list)
    
        st.write("### Choisir une opération...")
        #st.write("**ID Client est  :**", id_client)
        # .sidebar.radio
        #    ------------------------------------------------------------------------
        #    ------------------------------------------------------------------------        
        show_client_details = st.checkbox("Description des features" , value = False)
        ##show_client_suplemntaryinfo = st.checkbox("les informations supplémentaires")
        #show_credit_decision = st.checkbox("Décision de crédit")
        #show_credit_model = st.checkbox("Modèle de décision")
        #Evaluation_metric = st.checkbox("Métriques d'évaluation")
        show_metric_model = st.checkbox("Features importantes globale") 
        # show_client_comparison = st.checkbox("Comparer aux autres clients")
        shap_general = st.checkbox("Etude comparative Aux autres Clients")
        #visualisation_metric = st.checkbox("Visualisations")
        
    st.subheader("Diagramme à bandes")
    st.bar_chart(data)
    
    
    #histogram
    st.subheader("histogramme des 4 meilleures variables")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.DataFrame(data[:20], columns = ["AMT_ANNUITY","AMT_CREDIT","AMT_INCOME_TOTAL","AMT_GOODS_PRICE"])
    df.hist()
    plt.show()
    st.pyplot()
    
    #Line Chart (Graphique en ligne)
    st.subheader("Line Chart (Graphique en ligne des 4 meilleures variables)")
    st.line_chart(df)
    
    #Diagramme en batons
    st.subheader("Diagramme en batons")
    st.bar_chart(data["AMT_CREDIT"])
    st.bar_chart(data["AMT_INCOME_TOTAL"])
    #st.bar_chart(data["AMT_INCOME_TOTAL"])
    
    #Graphique en aires
    st.subheader("Graphique en aires Crédit/Revenu")
    chart_data = pd.DataFrame(data[:40], columns=["AMT_CREDIT","AMT_INCOME_TOTAL"])
    st.area_chart(chart_data)



if __name__ == '__main__':
    main()