import streamlit as st


def main():
	st.title('Ai que dlç')
	st.header('Thi is a leasson')
	st.subheader('I am in my hose')
	st.text('Fim da rotina')
	
	button = st.button('Clica')
	if button:
		st.text('Aí!')

	check = st.checkbox('CheckBox')
	if check:
		st.markdown('Aiiiiiin')

	radio = st.radio('Choose a option', ('option 1', 'option 2'))
	if radio == 'option 1':
		st.markdown('Option 1')
	elif radio == 'option 2':
		st.markdown('Option 2')

	select = st.selectbox('Select a option', ('Option 1', 'Option 2'))
	if select == 'Option 1':
		st.markdown('Option 1 selected')
	elif select == 'Option 2':
		st.markdown('Option 2 selected')

	multi_select = st.multiselect('Choose a option', ('Option 1', 'Option 2'))
	if multi_select == 'Option 1':
		st.markdown('Option 1 selected')
	elif multi_select == 'Option 2':
		st.markdown('Option 2 selected')

	up_file = st.file_uploader('Choose a file', type='csv')
	if up_file is not None:
		st.markdown('Arquivo não está vazio!')
	#st.image('igobarros.png')



if __name__ == '__main__':
	main()