import streamlit as st
from src.inference import get_prediction

#Initialise session state variable
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

def app_sidebar():
    st.sidebar.header('Passenger Details')
    gender_options = ['male', 'female']
    gender = st.sidebar.selectbox("Gender", gender_options)
    age = st.sidebar.text_input("Age")
    pclass_options = ['1st','2nd','3rd']
    pclass = st.sidebar.selectbox("Ticket class", pclass_options)
    embarked_options = ['Cherbourg','Southampton','Queenstown']
    embarked = st.sidebar.selectbox("Port of Embarkation", embarked_options)
    fare = st.sidebar.text_input("Passenger fare")
    sibsp = st.sidebar.slider("# of Siblings / Spouses Aboard", 0, 10, 0, 1)
    parch = st.sidebar.slider("# of Parents / Children Aboard", 0, 10, 0, 1)
    def get_input_features():
        input_features = {'gender': gender,
                          'age': float(age),
                          'pclass': pclass,
                          'embarked': embarked,
                          'fare': float(fare),
                          'sibsp': int(sibsp),
                          'parch': int(parch),
                         }
        return input_features
    sdb_col1, sdb_col2 = st.sidebar.columns(2)
    with sdb_col1:
        predict_button = st.sidebar.button("Assess", key="predict")
    with sdb_col2:
        reset_button = st.sidebar.button("Reset", key="clear")
    if predict_button:
        st.session_state['input_features'] = get_input_features()
    if reset_button:
        st.session_state['input_features'] = {}
    return None

def app_body():
    title = '<p style="font-family:arial, sans-serif; color:Black; font-size: 40px;"><b> Welcome to DSSI Titanic Survival Prediction</b></p>'
    st.markdown(title, unsafe_allow_html=True)
    default_msg = '**System assessment says:** {}'
    if st.session_state['input_features']:
        assessment = get_prediction(gender=st.session_state['input_features']['gender'],
                                    age=st.session_state['input_features']['age'],
                                    pclass=st.session_state['input_features']['pclass'],
                                    embarked=st.session_state['input_features']['embarked'],
                                    fare=st.session_state['input_features']['fare'],
                                    sibsp=st.session_state['input_features']['sibsp'],
                                    parch=st.session_state['input_features']['parch'])
        if assessment >= 0.5:
            st.success(default_msg.format('Survived'))
        else:
            st.warning(default_msg.format('Not Survived'))
    return None

def main():
    app_sidebar()
    app_body()
    return None

if __name__ == "__main__":
    main()