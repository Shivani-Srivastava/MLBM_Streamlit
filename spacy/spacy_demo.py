import streamlit as st
import spacy_streamlit
import pandas as pd
import time
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import Tree
import base64
import nltk
nltk.download('punkt')

try:
	nlp = spacy.load('en_core_web_sm')
except OSError:
	print('Downloading language model for the spaCy POS tagger\n',"(don't worry, this will only happen once)")
	from spacy.cli import download
	download('en_core_web_sm')
	nlp = spacy.load('en_core_web_sm')

def sent_tokenize1(raw_text):
	sents_list = sent_tokenize(raw_text)
	index0 = [i+1 for i in range(len(sents_list))]
	sents_pd=pd.DataFrame({'sl_num':index0, 'sentence':sents_list})
	return(sents_pd)

def token_attrib(sent0):
	doc = nlp(sent0)
	text=[]; lemma=[]; postag=[]; depcy=[]
	for token in doc:
		text.append(token.text)
		lemma.append(token.lemma_)
		postag.append(token.pos_)
		depcy.append(token.dep_)
	test_df = pd.DataFrame({'text':text, 'lemma':lemma, 'postag':postag, 'depcy':depcy})
	
	return(test_df)

def sent_attribs(sents_pd):
	token_df00 = pd.DataFrame(columns = ["doc_index", "sent_index", "text", "lemma", "postag", "depcy", "entity"])
	for i0 in range(sents_pd.shape[0]):
		tok_df = token_attrib(str(sents_pd.sentence.iloc[i0]))
		sent_index1 = [sents_pd.sl_num.iloc[i0]]*tok_df.shape[0]
		doc_index1 = [sents_pd.doc_index.iloc[i0]]*tok_df.shape[0]
		tok_df.insert(0, "sent_index", sent_index1)
		tok_df.insert(0, "doc_index", doc_index1)
		tok_df00 = pd.concat([tok_df00, tok_df])
	return(tok_df0)

def corpus2df(corpus0):
	df0 = pd.DataFrame(columns = ["doc_index", "sent_index", "text", "lemma", "postag", "depcy", "entity"])
	for i0 in range(corpus0.shape[0]):
		df1 = sent_tokenize1(corpus0.iloc[i0,0])
		doc_index0 = [i0+1]*len(df1)
		df1.insert(0, "doc_index", doc_index0)
		df2 = sent_attribs(df1)
		df0 = pd.concat([df0, df2])
	return(df0)

@st.cache(persist=True)
def processed_file(tok_df0):
	data1 = pd.DataFrame(tok_df0)
	return(data1)

def to_nltk_tree(node):
	if node.n_lefts + node.n_rights > 0:
		return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
	else:
		return node.orth_

def chunkAttrib(sent0):
	doc = nlp(sent0)
	chunk1 = [(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text) for chunk in doc.noun_chunks if len(word_tokenize(chunk.text)) > 1]
	out_df1 = pd.DataFrame(chunk1, columns = ['chText', 'chRootText', 'chRootDep', 'chRootHead'])
	return(out_df1)

def ner_bilou_tbl(docx):
	ent_token = [X for X in docx if X.ent_iob_ != 'O']
	ent_iob = [X.ent_iob_ for X in docx if X.ent_iob_ != 'O']
	ent_type = [X.ent_type_ for X in docx if X.ent_iob != 'O']
	ent_chunk = pd.DataFrame({'ent_token': ent_token, 'ent_iob':ent_iob, 'ent_type' : ent_type})
	return(ent_chunk)

def get_table_download_link(df):
	csv = df.to_csv(index = False)
	b64 = base64.b64encode(csv.encode()).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv"> Download dataframe as CSV </a>'
	return(href)

models = ["en_core_web_sm"]


def main():
	st.title("Spacy Demonstration")
	menu = ["Dataset Overview", "Annotate a File","Named Entity Recognition","Parsing Tree"]
	choice = st.sidebar.selectbox("Tabs :", menu)
	uploaded_file = st.sidebar.file_uploader("Upload a CSV/TXT file")
	data = pd.DataFrame(columns=['Upload', 'a', 'File'], index=range(2))
	

	if uploaded_file is not None:
		file_extension = uploaded_file.name.split(".")[-1]
		if file_extension == "csv":
			data = pd.read_csv(uploaded_file)
		elif file_extension == "txt":
			data = pd.read_csv(uploaded_file, sep = '\t', header=None, names = ['Text Data'])
		id_col = st.sidebar.selectbox('Select text column', options=data.columns)


	st.write('Scroll further to see tab outputs')
	
	if choice == "Dataset Overview":

		st.write("Top ten rows of the dataframe: ")
		st.markdown(' ### Data Sample')
		st.dataframe(data[:10])
		#st.write(data.iloc[1, data.columns.get_loc(id_col)])


	if choice == "Annotate a File":
		st.write("This displays annotated POSTagging for the uploaded file.")

		data0 = data[[id_col]]

		poso_df = pd.DataFrame(columns = ['sent_ind','text','lemma','postag','depcy'])
		
		t1 = time.time()
		for i in range(len(data0)):
			sent0 = str(data0.iloc[i, data0.columns.get_loc(id_col)])
			df0 = token_attrib(sent0)
			df0.insert(0,"sent_ind",i)
			poso_df = poso_df.append(df0)
			
		t2 = time.time()
		st.write(round(t2-t1,2)," seconds to process.")
		st.dataframe(poso_df)
	
	if choice == "Named Entity Recognition":
		st.write("This displays NER for the specific index in the uploaded file.")
		data0 = data[[id_col]]		
		row = st.sidebar.selectbox('Select row number', options=range(len(data0)))
		docu = str(data0.iloc[row, data0.columns.get_loc(id_col)])
		doc = spacy_streamlit.process_text("en_core_web_sm",docu)
		#st.write(doc.ents)
		#for ent in doc.ents:
			#st.write(str(data0.iloc[row]))
			#st.write(ent.text, ent.label_)
		spacy_streamlit.visualize_ner(doc)

	if choice == "Parsing Tree":
		st.write("This tab displays the dependency parse and part of speech tags")
		data0 = data[[id_col]]		
		row = st.sidebar.selectbox('Select row number', options=range(len(data0)))
		docu = str(data0.iloc[row, data0.columns.get_loc(id_col)])
		doc = spacy_streamlit.process_text("en_core_web_sm",docu)
		spacy_streamlit.visualize_parser(doc)
			
		

	
if __name__ == '__main__':
	main()