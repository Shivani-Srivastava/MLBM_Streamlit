import streamlit as st

from transformers import pipeline

summarizer = pipeline("summarization")

st.title('BERT: Summarization')

st.write('This is a demonstration of how BERT summarizes text.')

st.write('In case text has not been entered, the defaults are:')


st.write('Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital.')


sentence = st.text_area('Enter the text to be summarized.', value = 'Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital.')

#k = st.slider('Number of sentencest to generate', 1, 5)



output0 = summarizer(sentence)

#potential_mask_fills = []

#for i in range(len(output0)):
	#potential_mask_fills.append(output0[i]['generated_text'])

st.write(qn)
st.write(output0['summary_text'])


