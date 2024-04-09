from gensim.models.doc2vec import Doc2Vec
# from nltk.tokenize import word_tokenize
from numpy.linalg import norm
# import pandas as pd
import numpy as np
import PyPDF2 
import re
import plotly.graph_objects as go
import streamlit as st
import base64
from docx import Document
# from streamlit_pdf_viewer import pdf_viewer


st.set_page_config(layout="wide")
def main():
    st.title('Resume Scanner')   

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-color: black;
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    resume = ""
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file is not None:
        # Divide the page into two sections/columns
        col1, col2 = st.columns(2,gap="large")
       
        # Content for the left section (column)
        with col1:
            st.header("Uploaded resume")
        
            # Display uploaded Resume
           
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                pdf_bytes = uploaded_file.read()           
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                # pdf_viewer(pdf_bytes)
                pdf_display = F'<embed src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" type="application/pdf">'
                # # st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" type="application/pdf" style="border: none;"></iframe>', unsafe_allow_html=True)
                st.markdown(pdf_display, unsafe_allow_html=True)
                pdf = PyPDF2.PdfReader(uploaded_file)
                
                for i in range(len(pdf.pages)):
                    pageObj = pdf.pages[i]
                    resume += pageObj.extract_text()

            elif file_extension == 'docx':               
               content = read_word_docx(uploaded_file)
               st.text_area("Document Content:", value=content, height=475)
               resume = content

            # elif file_extension == 'txt':
            #     content = read_text_file(uploaded_file)
            #     st.markdown(f'<div style="word-wrap: break-word;overflow-y: scroll; max-height: 500px;">{content}</div>', unsafe_allow_html=True)
            #     resume = content
            else:
                st.warning("Unsupported file type. Please upload a PDF, Word document (docx), or text file.")

        with col2:
            st.header("Job description")
            user_input = st.text_area("Enter job description here:", height=475)

         # Centered button using CSS styling
        
        btn = st.button("Predict score",type="primary",use_container_width=True)   
        if btn:
            if user_input is not None:
                preprocess_Resume(resume,user_input)

def read_text_file(file):
    content = file.read()
    return content            

# def read_pdf(file_path):
#     with open(file_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page_num in range(len(pdf_reader.pages)):
#             text += pdf_reader.pages[page_num].extract_text()
#         return text

def read_word_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)
    
    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def preprocess_Resume(resume,jd):

    # pdf = PyPDF2.PdfReader(resumefile)
    # #pdf = PyPDF2.PdfReader('./CV/Jalpa-Dave.pdf')
    # resume = ""
    # for i in range(len(pdf.pages)):
    #     pageObj = pdf.pages[i]
    #     resume += pageObj.extract_text()

    # print(resume)

    # Apply to CV and JD
    input_CV = preprocess_text(resume)
    input_JD = preprocess_text(jd)


    # Model evaluation
    model = Doc2Vec.load('cv_job_maching.model')
    v1 = model.infer_vector(input_CV.split())
    v2 = model.infer_vector(input_JD.split())
    similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    print(round(similarity, 2))


    # Visualization
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = similarity,
        mode = "gauge+number",
        title = {'text': "Matching percentage (%)"},
        #delta = {'reference': 100},
        gauge = {
            'axis': {'range': [0, 100]},
            'steps' : [
                {'range': [0, 50], 'color': "#FFB6C1"},
                {'range': [50, 70], 'color': "#FFFFE0"},
                {'range': [70, 100], 'color': "#90EE90"}
            ],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}))

    fig.update_layout(width=600, height=400)  # Adjust the width and height as desired
    # fig.show()
    
    st.plotly_chart(fig,use_container_width=True)

    # Print notification
    if similarity < 50:        
        st.markdown("<h3 style='color:red; text-align: center;'>Low Match: The alignment between the resume and the job description is minimal.</h3>",unsafe_allow_html=True)        
    elif similarity >= 50 and similarity < 70:
        st.markdown("<h3 style='color:yellow; text-align: center;'>Moderate Match: There is a reasonable degree of alignment between the resume and the job description.</h3>",unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green; text-align: center;'>Great Match: The resume strongly aligns with the requirements and expectations outlined in the job description. </h3>",unsafe_allow_html=True)
        



if __name__ == "__main__":
    main()



    
    



