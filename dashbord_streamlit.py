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
st.text("Voici le Dashbord permettant de prédire le score d'un client et de comparer")
st.text("les informations relatives à un client ou un groupe de clients :")
#id_input = st.text_input("Veuillez saisir l'identifiant d'un client:", )


@st.cache
def load_data ():
    data = pd.read_parquet('app_test.parquet')
    data.drop(columns = {'Unnamed: 0'}  , inplace = True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(0, inplace=True)
    return data

data = load_data()

st.write(data.head(5))



# Affichage du titre et des données relatives au client sélectionné ------------
st.sidebar.header('Sélection du numéro du client')
id_client = st.sidebar.selectbox("Veuillez sélectionner l'identifiant d'un client:", data['SK_ID_CURR'])

st.subheader('Les données relatives au client sélectionné')

data_client = data.loc[data['SK_ID_CURR'] == int(id_client)]
#data_client.columns = ['Informations clients']
st.write(data_client)
    
################### C'est bon jusqu'ici ###############################################################    
    
def main():
    # local API (à remplacer par l'adresse de l'application déployée)
    # URL of the deployed api flask
    API_URL = "https://oumaar-application-programming.herokuapp.com/"

    ##################################
    # LIST OF API REQUEST FUNCTIONS

    # Obtenir la liste des SK_IDS (en cache)
    @st.cache
    def get_sk_id_list():
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"
        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of SK_IDS from the content
        SK_IDS = pd.Series(content['data']).values
        return SK_IDS

    # Obtenir des données personnelles (en cache)
    @st.cache
    def get_data_cust(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        PERSONAL_DATA_API_URL = API_URL + "data_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response to API request
        response = requests.get(PERSONAL_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        data_cust = pd.Series(content['data']).rename(select_sk_id)
        data_cust_proc = pd.Series(content['data_proc']).rename(select_sk_id)
        return data_cust, data_cust_proc

    # Obtenez des données de 20 voisins les plus proches dans le train (en cache)
    @st.cache
    def get_data_neigh(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        NEIGH_DATA_API_URL = API_URL + "neigh_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(NEIGH_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        X_neigh = pd.DataFrame(content['X_neigh'])
        y_neigh = pd.Series(content['y_neigh']['TARGET']).rename('TARGET')
        return X_neigh, y_neigh

    # Récupère toutes les données du train (en cache)
    @st.cache
    def get_all_proc_data_tr():
        # URL of the scoring API
        ALL_PROC_DATA_API_URL = API_URL + "all_proc_data_tr/"
        # save the response of API request
        response = requests.get(ALL_PROC_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        X_tr_proc = pd.DataFrame(content['X_tr_proc'])
        y_tr = pd.Series(content['y_train']['TARGET']).rename('TARGET')
        return X_tr_proc, y_tr

    # Obtenez le score d'un client candidat (en cache)
    @st.cache
    def get_cust_scoring(select_sk_id):
        # URL of the scoring API
        SCORING_API_URL = API_URL + "scoring_cust/?SK_ID_CURR=" + str(select_sk_id)
        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # getting the values from the content
        score = content['score']
        thresh = content['thresh']
        return score, thresh

    # Obtenir la liste des features
    @st.cache
    def get_features_descriptions():
        # URL of the aggregations API
        FEAT_DESC_API_URL = API_URL + "feat_desc"
        # Requesting the API and save the response
        response = requests.get(FEAT_DESC_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        features_desc = pd.Series(content['data']['Description']).rename("Description")
        return features_desc
    
    # Obtenir la liste des features importances (selon le modèle de classification lgbm)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        FEAT_IMP_API_URL = API_URL + "feat_imp"
        # Requesting the API and save the response
        response = requests.get(FEAT_IMP_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Obtenir les valeurs de forme du client et des 20 voisins les plus proches (en cache)
    @st.cache
    def get_shap_values(select_sk_id):
        # URL of the scoring API
        GET_SHAP_VAL_API_URL = API_URL + "shap_values/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(GET_SHAP_VAL_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame or pd.Series
        shap_val_df = pd.DataFrame(content['shap_val'])
        shap_val_trans = pd.Series(content['shap_val_cust_trans'])
        exp_value = content['exp_val']
        exp_value_trans = content['exp_val_trans']
        X_neigh_ = pd.DataFrame(content['X_neigh_'])
        return shap_val_df, shap_val_trans, exp_value, exp_value_trans, X_neigh_
  



        with st.sidebar:
        ## st.header(" Prêt à dépenser")
        st.write("## Identificateur du Client")
        #st.write("**ID Client est  :**", id_client)
        id_list = df["SK_ID_CURR"].tolist()
        # id_client = st.number_input("Sélectionner l'identifiant du client" , step = 1 , value = 100001 )
        id_client = st.number_input(" " , step = 1 , value = 100001 )
        # id_client = st.selectbox(
         #    "Sélectionner l'identifiant du client", id_list)

        st.write("## Choisir une opération")
        #st.write("**ID Client est  :**", id_client)
        # .sidebar.radio
        #    ------------------------------------------------------------------------
        #    ------------------------------------------------------------------------        
        show_client_details = st.checkbox("Informations fondamentales du client" , value = False)
        ##show_client_suplemntaryinfo = st.checkbox("les informations supplémentaires")
        show_credit_decision = st.checkbox("Décision de crédit")
        #show_credit_model = st.checkbox("Modèle de décision")
        #Evaluation_metric = st.checkbox("Métriques d'évaluation")
        show_metric_model = st.checkbox("Etude comparative Aux autres Clients")


        if (show_metric_model):
            st.header('‍👀 Comparaison aux autres clients')
            #st.subheader("Comparaison avec l'ensemble des clients")
            with st.expander("🔍 Explication de la comparaison faite"):
                st.write("Lorsqu'une variable est sélectionnée, un graphique montrant la distribution de cette variable selon la classe (remboursé ou défaillant) sur l'ensemble des clients (dont on connait l'état de remboursement de crédit) est affiché avec une matérialisation du positionnement du client actuel.") 

            with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
                var = st.selectbox("Sélectionner une variable",\
                                   list(personal_info_cols.values()))
                feature = list(personal_info_cols.keys())\
                [list(personal_info_cols.values()).index(var)]    

                if (feature in numerical_features):                
                    plot_distribution(data_train, feature, client_info[feature], var)   
                elif (feature in rotate_label):
                    univariate_categorical(data_train, feature, \
                                           client_info[feature], var, False, True)
                elif (feature in horizontal_layout):
                    univariate_categorical(data_train, feature, \
                                           client_info[feature], var, False, True, True)
                else:
                    univariate_categorical(data_train, feature, client_info[feature], var)

        #-------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        #-------------------------------------------------------        
        


if __name__ == '__main__':
    main()