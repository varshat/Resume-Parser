
# Resume Parser

## Overview

This is a resume parsing application designed to extract relevant information from resumes using the Word2Vec technique. The application utilizes the Word2Vec model to convert words into high-dimensional vectors, allowing for semantic analysis and similarity comparison.

## Features

- Parse resumes in various formats (e.g., PDF, DOCX) to extract key information such as skills, experiences, education, and contact details.
- Convert text data into Word2Vec embeddings to capture semantic meaning and relationships between words.
- Analyze resume content using Word2Vec similarity scores to identify relevant keywords, phrases, and context.
- Provide structured output of parsed resume data for further processing or integration with other applications.

## Installation

1. Clone the repository to your local machine:

   git clone https://github.com/varshat/Resume-Parser.git


### Install dependencies using pip:

pip install -r requirements.txt

The application can use a pre-trained Word2Vec model or train a new model on your dataset. If you prefer to use a pre-trained model, download it and place it in the models/ directory.

### Usage
Run the application: python streamlit run predic.py

Upload resumes: Upload resume files in supported formats (e.g., PDF, DOCX) using the provided interface.

### Parse resumes:

The application will process the uploaded resumes, extract relevant information, and display the parsed data.

### Analyze results:

Review the extracted information from the resumes, including skills, experiences, education, and contact details. Use the Word2Vec similarity scores to identify relevant keywords and phrases.

## DEMO



https://github.com/varshat/Resume-Parser/assets/7055503/c795b837-ce82-43c5-a819-f454c299a426


## Application link
https://resumescanner.streamlit.app/
