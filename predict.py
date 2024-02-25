from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2 
import re
import plotly.graph_objects as go
import streamlit as st
import base64
from docx import Document

st.set_page_config(layout="wide")
def main():
    st.title('Resume Scanner')
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
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
                st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" style="border: none;"></iframe>', unsafe_allow_html=True)

            # elif file_extension == 'docx':               
            #    content = read_word_docx(uploaded_file)
            #    st.markdown(f'<iframe src=https://docs.google.com/gview?url=http://writing.engr.psu.edu/workbooks/formal_report_template.doc&embedded=true, {content}" width="700" height="500" style="border: none;"></iframe>', unsafe_allow_html=True)
                
            # elif file_extension == 'txt':
            #     content = read_text(uploaded_file)
            # else:
            #     st.warning("Unsupported file type. Please upload a PDF, Word document (docx), or text file.")

            # st.text(content)
        with col2:
            st.header("Job description")
            user_input = st.text_area("Enter job description here:", height=475)

         # Centered button using CSS styling
        
        btn = st.button("Predict score",type="primary",use_container_width=True)   
        if btn:
            if user_input is not None:
                preprocess_Resume(uploaded_file,user_input)

            

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

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
   

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

def preprocess_Resume(resumefile,jd):

    pdf = PyPDF2.PdfReader(resumefile)
    #pdf = PyPDF2.PdfReader('./CV/Jalpa-Dave.pdf')
    resume = ""
    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()



    # JD by input text:
    # jd = input("Paste your JD here: ")

    # jd = """
    # • Develop statistical models for various predictive methods such as forecasting, classification,
    # clustering and regression.
    # • Analyze large datasets to provide strategic direction to clients for their business.
    # • Handling client issues related to ElegantJ BI tool (Plug &amp; play Predictive).
    # • Generating actionable insights from client data and creating presentations and dashboards to
    # make recommendations for improvement.
    # • Scripting using R language as well as Apache Spark + Java for predictive algorithms such as
    # forecasting, classification, clustering, association mining, regression, decision tree, correlation.
    # • Automating &amp; integrating Predictive algorithms - R scripts in Plug &amp; play Predictive
    # Analytics,
    # module of ElegantJ BI.
    # • Automating &amp; integrating Predictive algorithms - Spark scripts in Smarten, module of
    # ElegantJ
    # BI.
    # •  Preparing  and  conducting  demonstration  of  predictive  analytics  module  of  BI  along  with
    # marketing team.

    # """

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



    
    



