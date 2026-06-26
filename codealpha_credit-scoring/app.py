import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import random
import time


st.set_page_config(
    page_title="Smart Credit AI",
    page_icon="💳",
    layout="wide"
)



# ---------------- THEMES ----------------

c1,c2 = st.columns([10,1])

with c2:
    theme = st.selectbox(
        "🎨",
        [
            "Neon",
            "Ocean",
            "Purple",
            "Dark"
        ],
        label_visibility="collapsed"
    )



themes = {

"Neon":
{
"bg":"#020617",
"card":"#111827",
"glow":"#00ffff",
"accent":"#ff00ff"
},

"Ocean":
{
"bg":"#082f49",
"card":"#0c4a6e",
"glow":"#38bdf8",
"accent":"#22d3ee"
},

"Purple":
{
"bg":"#2e1065",
"card":"#581c87",
"glow":"#e879f9",
"accent":"#c084fc"
},

"Dark":
{
"bg":"#000000",
"card":"#1f2937",
"glow":"#8b5cf6",
"accent":"#6366f1"
}

}


t=themes[theme]



st.markdown(f"""

<style>

.stApp {{
background:
linear-gradient(
135deg,
{t["bg"]},
{t["card"]}
);
color:white;
}}


.card {{
background:{t["card"]};
padding:25px;
border-radius:20px;
margin:15px;
box-shadow:0 0 25px {t["glow"]};
}}


h1 {{
color:{t["glow"]};
text-align:center;
}}


h2 {{
color:{t["accent"]};
}}


.stButton>button {{
background:
linear-gradient(
90deg,
{t["accent"]},
{t["glow"]}
);
color:white;
border-radius:25px;
height:45px;
}}


div[data-testid="stMetric"] {{
background:{t["card"]};
padding:15px;
border-radius:15px;
box-shadow:0 0 15px {t["glow"]};
}}

</style>

""",
unsafe_allow_html=True)




# ---------------- MODEL ----------------

model=joblib.load("credit_model.pkl")
encoders=joblib.load("encoders.pkl")



def predict(data):

    cols=[
        "Sex",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose"
    ]

    for c in cols:
        data[c]=encoders[c].transform(data[c])


    prediction=model.predict(data)

    probability=model.predict_proba(data)

    score=probability[0][1]*100


    return prediction[0],score




# ---------------- HEADER ----------------


st.title("💳 Smart Credit Risk Intelligence")


st.markdown("""
<div class="card">

🤖 AI Powered Credit Scoring System

</div>
""",
unsafe_allow_html=True)



a,b,c=st.columns(3)

a.metric("Model","Random Forest")
b.metric("Accuracy","75%")
c.metric("Type","AI")



st.divider()



# ---------------- PREDICTION ----------------


st.header("🔍 Credit Prediction")



col1,col2=st.columns(2)



with col1:

    age=st.number_input(
        "Age",
        18,100,30
    )


    sex=st.selectbox(
        "Sex",
        ["male","female"]
    )


    job=st.selectbox(
        "Job",
        [0,1,2,3]
    )


    housing=st.selectbox(
        "Housing",
        ["own","free","rent"]
    )



with col2:

    saving=st.selectbox(
        "Saving Account",
        ["Unknown","little","moderate","rich","quite rich"]
    )


    checking=st.selectbox(
        "Checking Account",
        ["Unknown","little","moderate","rich"]
    )


    amount=st.number_input(
        "Credit Amount",
        100,100000,5000
    )


    duration=st.slider(
        "Duration",
        1,72,12
    )



purpose=st.selectbox(
"Purpose",
[
"radio/TV",
"education",
"furniture/equipment",
"car",
"business",
"domestic appliances",
"repairs",
"vacation/others"
]
)



if st.button("🚀 Predict Credit Risk"):


    data=pd.DataFrame({

        "Age":[age],
        "Sex":[sex],
        "Job":[job],
        "Housing":[housing],
        "Saving accounts":[saving],
        "Checking account":[checking],
        "Credit amount":[amount],
        "Duration":[duration],
        "Purpose":[purpose]

    })


    with st.spinner("🤖 AI Checking..."):

        time.sleep(2)

        result,score=predict(data)



    st.header("💯 Prediction Result")



    st.metric(
        "Credit Score",
        f"{score:.2f}%"
    )



    # GRAPH INSIDE PREDICTION

    st.subheader("📊 Prediction Confidence")


    fig,ax=plt.subplots(figsize=(4,2))


    ax.bar(
        ["Score"],
        [score]
    )

    ax.set_ylim(0,100)

    ax.set_ylabel("%")


    st.pyplot(fig)



    # BALLOONS EVERY PREDICTION

    st.balloons()



    if score>=80:

        st.success(
        "🟢 Low Risk Customer"
        )

    elif score>=60:

        st.warning(
        "🟡 Medium Risk Customer"
        )

    else:

        st.error(
        "🔴 High Risk Customer"
        )



st.divider()



# ---------------- INSIGHTS ----------------


st.header("🤖 AI Suggestions")


tips=[
"Maintain good repayment history",
"Lower loan duration improves safety",
"Stable housing improves score",
"Avoid very high credit amount"
]


st.info(
random.choice(tips)
)



st.divider()



# ---------------- REPORT ----------------


st.header("📄 Report")


st.download_button(
"📥 Download Report",
"""
Smart Credit Risk Report

Generated by AI Credit System

""",
"credit_report.txt"
)