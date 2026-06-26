import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="AI Character Predictor",
    page_icon="✍️",
    layout="centered"
)



# ---------------- THEMES ----------------


theme = st.selectbox(
    "🎨 Choose Theme",
    [
        "Dark Neon",
        "White Clean",
        "Blue AI"
    ]
)


if theme=="Dark Neon":

    bg="#050505"
    card="#151515"
    text="white"
    accent="#00ffff"


elif theme=="White Clean":

    bg="#f5f7fb"
    card="white"
    text="#111111"
    accent="#0066ff"


else:

    bg="#06162e"
    card="#102a52"
    text="white"
    accent="#00ff99"



st.markdown(
f"""
<style>

.stApp {{
background:{bg};
}}


h1 {{
color:{accent};
text-align:center;
font-size:45px;
font-weight:bold;
}}


h2,h3,p,label {{
color:{text} !important;
}}


.card {{

background:{card};

padding:25px;

border-radius:25px;

box-shadow:0px 0px 25px {accent};

}}



.stButton button {{

width:100%;
height:55px;

background:{accent};

color:black;

font-size:22px;

border-radius:20px;

font-weight:bold;

}}


</style>
""",
unsafe_allow_html=True
)




# ------------ LOAD MODEL ------------


model=tf.keras.models.load_model(
    "model.keras"
)



labels=[
'0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J','K','L','M',
'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'a','b','c','d','e','f','g','h','i','j','k','l','m',
'n','o','p','q','r','s','t','u','v','w','x','y','z'
]



# ------------ TITLE ------------


st.markdown(
"<h1>✍️ AI Handwritten Character Recognition</h1>",
unsafe_allow_html=True
)


st.markdown(
'<div class="card">',
unsafe_allow_html=True
)


st.subheader("🎨 Draw Character")


canvas=st_canvas(

fill_color="black",

stroke_color="white",

stroke_width=18,

background_color="black",

height=250,

width=250,

drawing_mode="freedraw",

key="canvas"

)



st.subheader("📤 Upload Image")


uploaded=st.file_uploader(
"Upload PNG / JPG",
type=["png","jpg","jpeg"]
)



img=None



if uploaded:

    image=Image.open(uploaded).convert("RGBA")

    st.image(
        image,
        width=200
    )

    img=np.array(image)



elif canvas.image_data is not None:

    img=canvas.image_data



st.markdown(
"</div>",
unsafe_allow_html=True
)





# ------------ PREDICT ------------


if st.button("🚀 Predict"):


    if img is None:

        st.warning(
            "Please draw or upload image"
        )


    else:


        img=cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_RGBA2GRAY
        )


        coords=cv2.findNonZero(img)


        if coords is not None:

            x,y,w,h=cv2.boundingRect(coords)

            img=img[y:y+h,x:x+w]



        img=cv2.resize(
            img,
            (28,28)
        )


        img=cv2.rotate(
            img,
            cv2.ROTATE_90_CLOCKWISE
        )


        img=cv2.flip(
            img,
            1
        )


        img=img/255.0


        img=img.reshape(
            1,28,28,1
        )


        result=model.predict(img)


        index=np.argmax(result)

        confidence=np.max(result)*100



        st.balloons()



        st.markdown(
        f"""

        <div class="card">

        <h2 style="text-align:center;">
        Prediction Result
        </h2>


        <h1 style="text-align:center;">
        {labels[index]}
        </h1>


        <h3 style="text-align:center;">
        Confidence : {confidence:.2f} %
        </h3>


        </div>

        """,
        unsafe_allow_html=True
        )