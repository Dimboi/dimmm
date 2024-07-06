import streamlit as st
import pickle
import pyreadstat

sakit_jantung = pickle.load(open('sakit_jantung.sav', 'rb'))


st.title('chance orang idup pas kena jantung(1 untuk hidup/0 untuk dikubur)')

age = st.number_input ('umur anda: ')
anaemia = st.number_input ('apakah anda anemia: ')
creatinine_phosphokinase = st.number_input ('kandungan ceratine anda: ')
diabetes = st.number_input ('apakah anda diabetes: ')
ejection_fraction = st.number_input ('presentase darah yang dipompa keluar dari ventrikel kiri setiap jantung berdetak: ')
high_blood_pressure = st.number_input ('apakah anda memiliki tekanan darah tinggi?(1/0)')
platelets = st.number_input ('trombosit dalam darah(kiloplatelets/ml): ')
serum_creatinine = st.number_input ('kadar serum ceratinine dalam darah(mg/dl): ')
serum_sodium = st.number_input ('kadar natriumdalam darah(mEq/L): ')
sex = st.number_input ('jenis kelamin(1=pria, 0=wanita): ')
smoking = st.number_input ('apakah anda perokok?(1/0): ')
time = st.number_input ('berapa hari lalu terakhir kali anda melakukan pengecekan: ')


diagnosis = ''

if st.button('Gacha Kematian'):
    features = [[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]]
    diagnosis = sakit_jantung.pre(features)

    if diagnosis[0] == 1:
        diagnosis = 'Pasien meninggal'
    else:
        diagnosis = 'Pasien hidup'

    st.success(diagnosis)