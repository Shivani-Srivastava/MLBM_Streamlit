import streamlit as st

from transformers import pipeline

generator = pipeline("text-generation")


st.title('BERT: Text Generation')

st.write('This is a demonstration of how BERT uses a pipeline to generate text, based on given prompts.')

st.write('In case text has not been entered, the default sentence is:')

st.write('In this three course meal, you will be served')

#st.write('Note that inserting <mask> in the sentence is important for the transformer to know where to fill the word.')

sentence = st.text_area('Enter the prompt tp generate text', value = 'In this three course meal, you will be served')
k = st.slider('Number of sentencest to generate', 1, 5)


output0 = generator(sentence, num_return_sequences = k)

potential_mask_fills = []

for i in range(len(output0)):
	potential_mask_fills.append(output0[i]['generated_text'])


st.write(potential_mask_fills)


