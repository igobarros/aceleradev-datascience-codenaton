import pandas as pd
import streamlit as st
import base64



def get_table_download_link(df):
	"""Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
	csv = df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode() # some strings <-> bytes conversions necessary here
	href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
	return href


def main():
	st.image('logo.png', width=200)
	st.title('Aceleradev Data Sciencee')
	st.subheader('Semana 2 - Pré-processamento de dados com python')
	st.image('https://media.giphy.com/media/KyBX9ektgXWve/giphy.gif', width=200)
	file = st.file_uploader('Informe a base de análise (.csv)', type='csv')

	if file is not None:
		st.subheader('Analisando os dados')
		df = pd.read_csv(file)

		st.markdown('**Número de linhas**')
		st.markdown(df.shape[0])
		st.markdown('**Número de colunas**')
		st.markdown(df.shape[1])

		st.markdown('**Visualizando o dataframe**')
		value = st.slider('Informe o número de linhas do dataframe', min_value=1, max_value=1000)
		st.dataframe(df.head(value))

		st.markdown('**Nome das colunas**')
		st.markdown(list(df.columns))

		exploration = pd.DataFrame({
				'names': df.columns
				, 'types': df.dtypes
				, 'NA #': df.isna().sum()
				, 'NA %': (df.isna().sum() / df.shape[0]) * 100
			})

		st.markdown('**Contagem dos tipos de valores**')
		st.write(exploration.dtypes.value_counts())

		st.markdown('**Nomes das colunas do tipo int**')
		st.markdown(list(exploration[exploration['types'] == 'int64']['names']))

		st.markdown('**Nomes das colunas do tipo float**')
		st.markdown(list(exploration[exploration['types'] == 'float64']['names']))

		st.markdown('**Nomes das colunas do tipo object**')
		st.markdown(list(exploration[exploration['types'] == 'object']['names']))

		st.markdown('**Tabela com coluna e percentual de valores faltantes**')
		st.table(exploration[exploration['NA #'] != 0][['types', 'NA %']])

		st.subheader('Inputação de dados numéricos')
		percentual = st.slider('Informe o limite de percentual faltante de limite para as colunas você deseja inputar os dados', min_value=0, max_value=100)
		column_list = list(exploration[exploration['NA %'] < percentual]['names'])

		select_method = st.radio('Informe o método', ('Média', 'Mediana'))
		st.markdown('Você selecionou ' + str(select_method))

		if select_method == 'Média':
			df_inputed = df[column_list].fillna(df[column_list].mean())
			exploration_inputed = pd.DataFrame({
					'names': df_inputed.columns
					, 'types': df_inputed.dtypes
					, 'NA #': df_inputed.isna().sum()
					, 'NA %': (df_inputed.isna().sum() / df_inputed.shape[0]) * 100
				})
			st.table(exploration_inputed[exploration_inputed['types'] != 'object']['NA %'])
			st.subheader('Dados Inputados! Faça o download abaixo: ')
			st.markdown(get_table_download_link(df_inputed), unsafe_allow_html=True)

		elif select_method == 'Mediana':
			df_inputed = df[column_list].fillna(df[column_list].median())
			exploration_inputed = pd.DataFrame({
					'names': df_inputed.columns
					, 'types': df_inputed.dtypes
					, 'NA #': df_inputed.isna().sum()
					, 'NA %': (df_inputed.isna().sum() / df_inputed.shape[0]) * 100
				})
			st.table(exploration_inputed[exploration_inputed['types'] != 'object']['NA %'])
			st.subheader('Dados Inputados! Faça o download abaixo: ')
			st.markdown(get_table_download_link(df_inputed), unsafe_allow_html=True)



if __name__ == '__main__':
	main()