import streamlit as st

from transformers import pipeline

question_answerer = pipeline("question-answering")


st.title('BERT: Context Driven QA')

st.write('This is a demonstration of how BERT uses a pipeline to answer questions, based on user provided context.')

st.write('In case text has not been entered, the defaults are:')

st.write('Question: Where do you study?')
st.write('Context: My name is <mask> and I am studying Business at Indian School of Business.')

#st.write('Note that inserting <mask> in the sentence is important for the transformer to know where to fill the word.')

cntxt = st.text_area('Enter the context.', value = 'My name is <mask> and I am studying Business at Indian School of Business.')
qn = st.text_area('Enter the question.', value = 'Where do you study?')
#k = st.slider('Number of sentencest to generate', 1, 5)


#output0 = generator(sentence, num_return_sequences = k)

output0 = question_answerer(question = qn, context=cntxt)

#potential_mask_fills = []

#for i in range(len(output0)):
	#potential_mask_fills.append(output0[i]['generated_text'])

st.write(qn)
st.write(output0['answer'])


