import streamlit as st

from transformers import pipeline

unmasker = pipeline("fill-mask")

#col2, col3 = st.columns([6,1])

st.title('BERT: Fill Masks Demo')

st.write('This is a demonstration of how BERT uses a pipeline to fill <mask>-d position in sentences with words they have most frequently been associated with')

st.write('In case text has not been entered, the default sentence is:')

st.write('I like listening to songs by artists like <mask> and Beethoven.')

st.write('Note that inserting <mask> in the sentence is important for the transformer to know where to fill the word.')

sentence = st.text_area('Enter the sentence with <mask> in place of the variable word.', value = 'I like viewing art by artists like <mask> and da Vinci.')
k = st.slider('Pick a number', 1, 20)


output0 = unmasker(sentence, top_k = k)

potential_mask_fills = []

for i in range(len(output0)):
	potential_mask_fills.append(output0[i]['sequence'])


st.write(potential_mask_fills)


