import streamlit as st

from transformers import pipeline

summarizer = pipeline("summarization")

st.title('BERT: Summarization')

st.write('This is a demonstration of how BERT summarizes text.')

st.write('In case text has not been entered, the defaults are:')


st.write('Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital.')


sentence = st.text_area('Enter the text to be summarized.', value = 'Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital.')

output0 = summarizer(sentence)

st.write(output0[0]['summary_text'])


