import streamlit as st
import streamlit_tags as sttags
import numpy as np
import joblib

# Preset values
FF_OPTIONS = ["Stable", "Volatile", "Hightly Volatile", "Very Highly Volatile"]
FF_TEAM_COMP = {"Stable":1, 
                "Volatile":0.98, 
                "Hightly Volatile":0.95, 
                "Very Highly Volatile":0.91}
FF_PROCESS = {"Stable":1, 
              "Volatile":0.98, 
              "Hightly Volatile":0.94, 
              "Very Highly Volatile":0.89}
FF_ENV_FACT = {"Stable":1, 
               "Volatile":0.99, 
               "Hightly Volatile":0.98, 
               "Very Highly Volatile":0.96}
FF_TEAM_DYN = {"Stable":1, 
               "Volatile":0.98, 
               "Hightly Volatile":0.91, 
               "Very Highly Volatile":0.85}

DF_OPTIONS = ["Normal", "High", "Very High", "Extra High"]

DF_TEAM_CHANGE = {"Normal":1, 
                  "High":0.98, 
                  "Very High":0.95, 
                  "Extra High":0.91}

DF_NEW_TOOL = {"Normal":1, 
                "High":0.99, 
                "Very High":0.97, 
                "Extra High":0.96}

DF_VENDOR_DEFECT = {"Normal":1, 
                    "High":0.98, 
                    "Very High":0.94, 
                    "Extra High":0.9}

DF_TEAM_RESP = {"Normal":1, 
                "High":0.99, 
                "Very High":0.98, 
                "Extra High":0.98}

DF_PERSONAL_ISSUE = {"Normal":1, 
                     "High":0.99, 
                     "Very High":0.99, 
                     "Extra High":0.98}

DF_STAKEHOLDER = {"Normal":1, 
                  "High":0.99, 
                  "Very High":0.98, 
                  "Extra High":0.96}                     

DF_UNCLEAR_REQ = {"Normal":1, 
                  "High":0.98, 
                  "Very High":0.97, 
                  "Extra High":0.95}

DF_CHANGE_REQ = {"Normal":1, 
                 "High":0.99, 
                 "Very High":0.98, 
                 "Extra High":0.97}                  

DF_RELOCATE = {"Normal":1, 
               "High":0.99, 
               "Very High":0.99, 
               "Extra High":0.98}

TRAD_MODEL_NAME = "trad_trained_model.joblib"
AGILE_MODEL_NAME = "agile_trained_model.joblib"



def callback():
    """
    Changes the current state to True if a button is clicked
    """
    st.session_state.button_clicked = True


@st.cache(allow_output_mutation=True)
def load_trad_model():
    """
    Load trad model
    """
    trad_model = joblib.load(TRAD_MODEL_NAME)
    return trad_model


@st.cache(allow_output_mutation=True)
def load_agile_model():
    """
Load agile model
    """
    agile_model = joblib.load(AGILE_MODEL_NAME)
    return agile_model


def trad_predict_effort(model, numpy_arr):
    """
    Predicts the effort required for traditional projects
    """
    effort = model.predict(numpy_arr)
    return int(effort)


def predict_sp(story_size_lst, complexity_lst):
    """
    Computes the total story points for all user stories
    """
    sp = []
    for num1, num2 in zip(story_size_lst, complexity_lst):
        sp.append(num1*num2)
    return sum(sp)


def agile_predict_effort(model, numpy_arr, workdays):
    """
    Predicts the effort required for agile projects
    """
    effort = model.predict(numpy_arr)
    effort = int(effort) * (1/workdays)
    return round(effort, 1)


if __name__ == "__main__":
    submitted1 = False
    submitted2 = False

    # check if button is clicked
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False

    st.set_page_config(page_title="MCS17", layout="wide")
    st.title("Software Effort Estimation Web App")
    method_select = st.selectbox("Select project development methodology", options=["Agile", "Traditional"])

    if method_select == "Agile":
        # task titles
        task_title_lst = sttags.st_tags(label="Enter task title", text="Press enter to add more")
        # if proceed has not been clicked yet 
        if st.button("Proceed", on_click=callback) or st.session_state.button_clicked:
            # prompt user error
            if not task_title_lst:
                st.error("Please enter a task title!")
                st.stop()
            # fill in form
            with st.form("form1"):
                c1, c2 = st.columns(2)
                story_size_lst = []
                complexity_lst = []
                # generate user stories based on the no of task titles entered
                for i, x in enumerate(task_title_lst):
                    with st.container():
                        with c1:
                            st.write("Task ", i+1, ": ", task_title_lst[i])
                            story_size_lst.append(st.slider("Rate story size", min_value=1, max_value=5, step=1, key="story_%d" % i))
                        with c2:
                            st.write(".")
                            complexity_lst.append(st.slider("Rate user story complexity", min_value=1, max_value=5, step=1, key="comp_%d" % i))

                team_vel = st.number_input('Enter team velocity', min_value=0, step=1)
                sprint_size = st.number_input('Enter number of days in sprint', min_value=1, max_value=28, step=1)
                work_days = st.number_input('Enter work days per month', min_value=1, max_value=31, step=1)
                c1, c2, c3 = st.columns(3)
                with st.container():
                    with c1:
                        ff_team_composition = st.select_slider("Rate team composition", options=FF_OPTIONS)
                    with c2:    
                        ff_process = st.select_slider("Rate development process", options=FF_OPTIONS)
                    with c3:
                        ff_environment_fact = st.select_slider("Rate environmental factors", options=FF_OPTIONS)
                with st.container():
                    with c1:
                        ff_team_dynamic = st.select_slider("Rate team dynamics", options=FF_OPTIONS)
                    with c2:
                        df_team_change = st.select_slider("Rate expected team changes", options=DF_OPTIONS)
                    with c3:
                        df_new_tool = st.select_slider("Rate introduction to new tools", options=DF_OPTIONS)
                with st.container():
                    with c1:
                        df_vendor_defect = st.select_slider("Rate vendor's defect", options=DF_OPTIONS)
                    with c2:
                        df_team_responsibility = st.select_slider("Rate team member's responsibilities", options=DF_OPTIONS)
                    with c3:
                        df_personal_issue = st.select_slider("Rate personal issues", options=DF_OPTIONS)
                with st.container():
                    with c1:
                        df_stakeholder = st.select_slider("Rate stakeholder", options=DF_OPTIONS)
                    with c2:
                        df_unclear_requirements = st.select_slider("Rate unclear requirements", options=DF_OPTIONS)
                    with c3:
                        df_change_requirements = st.select_slider("Rate changing requirements", options=DF_OPTIONS)
                with st.container():
                    with c1:
                        df_relocation = st.select_slider("Rate relocation", options=DF_OPTIONS)
                submitted1 = st.form_submit_button("Predict Software Effort")

            if submitted1:
                # load model
                with st.spinner("Loading model from the server, this may take a while..."):
                    agile_model = load_agile_model()
                    st.success('Model Loaded Successfully!')
                # compute friction factors
                ff_score = FF_TEAM_COMP[ff_team_composition]*FF_ENV_FACT[ff_environment_fact]*FF_PROCESS[ff_process]*FF_TEAM_DYN[ff_team_dynamic]
                # compute dynamic forces factors
                df_score = DF_TEAM_CHANGE[df_team_change]*DF_NEW_TOOL[df_new_tool]*DF_CHANGE_REQ[df_change_requirements]*DF_PERSONAL_ISSUE[df_personal_issue]*DF_RELOCATE[df_relocation]*DF_STAKEHOLDER[df_stakeholder]*DF_TEAM_RESP[df_team_responsibility]*DF_UNCLEAR_REQ[df_unclear_requirements]*DF_VENDOR_DEFECT[df_vendor_defect]
                deceleration = ff_score*df_score
                final_vel = team_vel ** deceleration
                initial_vel = team_vel / sprint_size
                total_sp = predict_sp(story_size_lst, complexity_lst)
                # store all input fields into a numpy array
                user_input = np.array([[total_sp, initial_vel, deceleration, final_vel, sprint_size, work_days]])
                # predict the effort using the loaded model and user inputs
                effort = agile_predict_effort(agile_model, user_input, work_days)
                st.balloons()
                st.write("Suggested effort: ", effort, " months")

    elif method_select == "Traditional":
        with st.form("form2"):
            c1, c2 = st.columns(2)
            with st.container():
                with c1:
                    team_exp = st.number_input('Enter team experience mesured in years (-1 for zero experience)', min_value=-1, value=0, step=1)
                with c2:
                    manager_exp = st.number_input('Enter manager experience mesured in years (-1 for zero experience)', min_value=-1, value=0, step=1)
            with st.container():
                with c1:
                    length = st.number_input('Enter duration of the project in months', min_value=0, step=1)
                with c2:
                    trans = st.number_input('Enter number of basic logical transactions in the system', min_value=0, step=1)
            with st.container():
                with c1:
                    entity = st.number_input('Enter number of entities in the systems data model', min_value=0, step=1)
                with c2:
                    points_non_adjust = st.number_input('Enter the size of the project measured in adjusted function points.', min_value=0, step=1)
            submitted2 = st.form_submit_button("Predict Software Effort")

        if submitted2:
            # load model
            with st.spinner("Loading model from the server, this may take a while..."):
                trad_model = load_trad_model()
                st.success('Model Loaded Successfully!')
            # store all input fields into a numpy array
            user_input = np.array([[team_exp, manager_exp, length, trans, entity, points_non_adjust]])
            # predict the effort using the loaded model and user inputs
            effort = trad_predict_effort(trad_model, user_input)
            st.balloons()
            st.write("Suggested effort: ", effort, " person-hours")




    
