import streamlit as st
import pandas as pd
import altair as alt


def histogram(column, df):
    chart = alt.Chart(df, width=600).mark_bar().encode(
        alt.X(column, bin=True),
        y='count()', tooltip=[column, 'count()']
    ).interactive()
    return chart


def barplot(num_column, cat_column, df):
    bars = alt.Chart(df, width=600).mark_bar().encode(
        x=alt.X(num_column, stack='zero'),
        y=alt.Y(cat_column),
        tooltip=[cat_column, num_column]
    ).interactive()
    return bars


def boxplot(num_column, cat_column, df):
    boxplot_ = alt.Chart(df, width=600).mark_boxplot().encode(
        x=num_column,
        y=cat_column
    )
    return boxplot_


def scatterplot(x, y, color, df):
    scatter = alt.Chart(df, width=800, height=400).mark_circle().encode(
        alt.X(x),
        alt.Y(y),
        color=color,
        tooltip=[x, y]
    ).interactive()
    return scatter


def correlationplot(df, num_columns):
    cor_data = (df[num_columns]).corr().stack().reset_index().rename(
        columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'})
    cor_data['correlation_label'] = cor_data['correlation'].map(
        '{:.2f}'.format)  # Round to 2 decimal
    base = alt.Chart(cor_data, width=500, height=500).encode(
        x='variable2:O', y='variable:O')
    text = base.mark_text().encode(text='correlation_label', color=alt.condition(alt.datum.correlation > 0.5, alt.value('white'),
                                                                                 alt.value('black')))

# The correlation heatmap itself
    cor_plot = base.mark_rect().encode(
        color='correlation:Q')

    return cor_plot + text


def main():

    st.image('logo.png', width=200)
    st.title('AceleraDev Data Science')
    st.subheader('Semana 3 - Análise de dados exploratória')
    st.image('https://media.giphy.com/media/R8bcfuGTZONyw/giphy.gif', width=200)

    file = st.file_uploader(
        'Escolha a base de dados que deseja analisar (.csv)', type='csv')

    if file is not None:

        st.subheader('Estatística descritiva univariada')

        df = pd.read_csv(file)

        aux = pd.DataFrame({"colunas": df.columns, 'tipos': df.dtypes})

        num_columns = list(aux[aux['tipos'] != 'object']['colunas'])
        cat_columns = list(aux[aux['tipos'] == 'object']['colunas'])
        columns = list(df.columns)

        col = st.selectbox('Selecione a coluna :', num_columns)

        if col is not None:
            st.markdown('Selecione o que deseja analisar :')

            is_mean = st.checkbox('Média')
            if is_mean:
                st.markdown(df[col].mean())

            is_median = st.checkbox('Mediana')
            if is_median:
                st.markdown(df[col].median())

            is_std = st.checkbox('Desvio padrão')
            if is_std:
                st.markdown(df[col].std())

            is_kurtosis = st.checkbox('Kurtosis')
            if is_kurtosis:
                st.markdown(df[col].kurtosis())

            is_skewness = st.checkbox('Skewness')
            if is_skewness:
                st.markdown(df[col].skew())

            is_describe = st.checkbox('Describe')
            if is_describe:
                st.table(df[num_columns].describe().transpose())

        st.subheader('Visualização dos dados')
        st.image(
            'https://media.giphy.com/media/Rkoat5KMaw2aOHDduz/giphy.gif', width=200)
        st.markdown('Selecione a visualizacao')

        is_hist = st.checkbox('Histograma')

        if is_hist:
            col_num = st.selectbox(
                'Selecione a Coluna Numerica: ', num_columns, key='unique')

            st.markdown('Histograma da coluna : ' + str(col_num))
            st.write(histogram(col_num, df))

        is_bars = st.checkbox('Gráfico de barras')

        if is_bars:
            col_num_bars = st.selectbox(
                'Selecione a coluna numerica: ', num_columns, key='unique')
            col_cat_bars = st.selectbox(
                'Selecione uma coluna categorica : ', cat_columns, key='unique')

            st.markdown('Gráfico de barras da coluna ' +
                        str(col_cat_bars) + ' pela coluna ' + col_num_bars)
            st.write(barplot(col_num_bars, col_cat_bars, df))

        is_boxplot = st.checkbox('Boxplot')

        if is_boxplot:
            col_num_box = st.selectbox(
                'Selecione a Coluna Numerica:', num_columns, key='unique')
            col_cat_box = st.selectbox(
                'Selecione uma coluna categorica : ', cat_columns, key='unique')

            st.markdown('Boxplot ' + str(col_cat_box) +
                        ' pela coluna ' + col_num_box)
            st.write(boxplot(col_num_box, col_cat_box, df))

        is_scatter = st.checkbox('Scatterplot')

        if is_scatter:
            col_num_x = st.selectbox(
                'Selecione o valor de x ', num_columns, key='unique')
            col_num_y = st.selectbox(
                'Selecione o valor de y ', num_columns, key='unique')

            col_color = st.selectbox('Selecione a coluna para cor', columns)
            st.markdown('Selecione os valores de x e y')
            st.write(scatterplot(col_num_x, col_num_y, col_color, df))

        is_correlation = st.checkbox('Correlacao')

        if is_correlation:
            st.markdown('Gráfico de correlação das colunas númericas')
            st.write(correlationplot(df, num_columns))


if __name__ == '__main__':
    main()
