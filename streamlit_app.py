import streamlit as st
import pickle

sakit_jantung = pickle.load(open('sakit_jantung.sav', 'rb'))

st.title('chance orang idup pas kena jantung(yes/no)')
age = st.text_input ('umur anda: ')
anemia = st.text_input ('apakah anda anemia: ')
ceratine_phosphokinase = st.text_input ('kandungan ceratine anda: ')
diabetes = st.text_input ('kandungan diabetes anda: ')

diagnosis = ''

if st.button ('gacha kematian: ')
diagnosis_pred = sakit_jantung.predict([[age, anemia, ceratine_phosphokinase, diabetes]])

  if (diagnosis_pred[0] == 1):
    diagnosis = 'pasien meninggal'
  else:
    diagnosis = 'pasien hidup'

  st.success(diagnosis)
