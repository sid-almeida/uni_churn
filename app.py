import streamlit as st
import pandas as pd
import os
import sweetviz as sv
import codecs
import streamlit.components.v1 as components
import pickle as pkl

def st_display_sweetviz(report_html,width=1500,height=1000):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)

with st.sidebar:
    st.image("https://github.com/sid-almeida/uni_churn/blob/main/Brainize%20Tech(1).png?raw=true", width=250)
    st.title("UniChurn")
    st.write('---')
    choice = st.radio("**Navegação:**", ("Upload", "Análise", "Machine Learning", "Previsão de Conjunto"))
    st.info("Esta aplicação permite a análise de dados de uma universidade fictícia, com o objetivo de prever a evasão de alunos."
            " Além disso, ela utiliza Machine Learning para prever o estado futuro de alunos.")
    st.write('---')

if os.path.exists("data.csv"):
    dataframe = pd.read_csv("data.csv")

if choice == "Upload":
    st.subheader("Upload de dados (Treino / Teste)")
    st.write("Faça o upload do arquivo .csv para análise e modelagem.")
    st.write('---')
    file = st.file_uploader("Upload do arquivo", type=["csv"])
    if file is not None:
        datarame = pd.read_csv(file)
        st.dataframe(dataframe.head(10))
        st.success("Upload realizado com sucesso!")
        st.markdown("##")
        dataframe.to_csv("data.csv", index=False)
        st.success("Arquivo salvo com sucesso!")
        # se o dataframe tiver muitas colunas categóricas, criar outro data_num e transformar em numéricas com o LabelEncoder
        if len(dataframe.select_dtypes(include="object").columns) > 0:
            data_num = dataframe.copy()
            from sklearn.preprocessing import LabelEncoder
            for col in data_num.select_dtypes(include="object").columns:
                le = LabelEncoder()
                data_num[col] = le.fit_transform(data_num[col])
            data_num.to_csv("data.csv", index=False)
        else:
            pass
    else:
        st.warning("Por favor, faça o upload do arquivo .csv.")

if choice == "Análise":
    st.subheader("Análise de dados (SweetViz)")
    st.write("Análise exploratória dos dados com Sweetviz.")
    st.write('---')
    if os.path.exists("data.csv"):
        dataframe = pd.read_csv("data.csv")
        if dataframe is not None:
            report = sv.analyze(dataframe)
            # st.write(report.show_html(), unsafe_allow_html=True)
            st.success("Análise realizada com sucesso!")
            report.show_html(open_browser=False)
            with open("SWEETVIZ_REPORT.html", "w") as html_file:
                html_file.write(report._page_html)
            st_display_sweetviz("SWEETVIZ_REPORT.html")
            st.write('---')
        else:
            st.write('---')
            st.warning("Por favor, faça o upload do arquivo .csv.")
            st.write('---')
    else:
        st.write('---')
        st.warning("Por favor, faça o upload do arquivo .csv.")
        st.write('---')

if choice == "Machine Learning":
    st.subheader("Treino do Modelo (Treino / Avaliação)")
    st.write("Selecione o augorítmo para ser utilizado.")
    st.write('---')
    if os.path.exists("data.csv"):
        dataframe = pd.read_csv("data.csv")
        st.header("Treino de modelos de Machine Learning")
        st.subheader("Treino de modelos de Machine Learning para prever a evasão de alunos.")
        st.write('---')
        problema = st.selectbox("Selecione o problema:", ("Classificação", "Regressão"))
        st.write('---')
        if problema == "Classificação":
            modelo = st.selectbox("Selecione o modelo:", ("Logistic Regression", "Random Forest", "XGBoost"))
        if problema == "Regressão":
            modelo = st.selectbox("Selecione o modelo:", ("Linear Regression", "Random Forest", "XGBoost"))
            if modelo == "Linear Regression":
                st.warning("AVISO: Este modelo não é adequado para o problema de classificação!")
                # selectbox para selecionar o alvo
                alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
                # botão para treinar o modelo de regressão linear
                st.button("Treinar modelo de Regressão linear")
                if st.button:
                    # treinando o modelo
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import mean_squared_error
                    X = dataframe.drop('STATUS', axis=1)
                    y = dataframe['STATUS']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.subheader("**Avaliação do Modelo**")
                    st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
                    st.write("R2:", model.score(X_test, y_test))
                    st.success("Modelo treinado com sucesso!")
                # botão para salvar o modelo
                st.button("Salvar modelo")
                if st.button:
                    import pickle
                    pickle.dump(model, open("model.pkl", "wb"))
                    st.success("Modelo salvo com sucesso!")
        if modelo == "Logistic Regression":
            st.warning("Este modelo não é adequado para o problema de regressão.")
            # selectbox para selecionar o alvo
            alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
            # botão para treinar o modelo de regressão logística
            st.button("Treinar modelo de Regressão logística")
            if st.button:
                # treinando o modelo
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                X = dataframe.drop('STATUS', axis=1)
                y = dataframe['STATUS']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("**Avaliação do Modelo**")
                st.write("Acurácia:", accuracy_score(y_test, y_pred))
                st.success("Modelo treinado com sucesso!")
            # botão para salvar o modelo
            st.button("Salvar modelo")
            if st.button:
                import pickle
                pickle.dump(model, open("model.pkl", "wb"))
                st.success("Modelo salvo com sucesso!")
        elif modelo == "Random Forest":
            # selectbox para selecionar o alvo
            alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
            # botão para treinar o modelo de random forest
            st.button("Treinar modelo de Random Forest")
            if st.button:
                # treinando o modelo
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                X = dataframe.drop('STATUS', axis=1)
                y = dataframe['STATUS']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("**Avaliação do Modelo**")
                st.write("Acurácia:", accuracy_score(y_test, y_pred))
                st.success("Modelo treinado com sucesso!")
            # botão para salvar o modelo
            st.button("Salvar modelo")
            if st.button:
                import pickle
                pickle.dump(model, open("model.pkl", "wb"))
                st.success("Modelo salvo com sucesso!")
        elif modelo == "XGBoost":
            #selectbox para selecionar o alvo
            alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
            # botão para treinar o modelo de xgboost
            st.button("Treinar modelo de XGBoost")
            if st.button:
                # treinando o modelo
                from xgboost import XGBClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                X = dataframe.drop('STATUS', axis=1)
                y = dataframe['STATUS']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = XGBClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("**Avaliação do Modelo**")
                st.write("Acurácia:", accuracy_score(y_test, y_pred))
                st.success("Modelo treinado com sucesso!")
            # botão para salvar o modelo
            st.button("Salvar modelo")
            if st.button:
                import pickle
                pickle.dump(model, open("model.pkl", "wb"))
                st.success("Modelo salvo com sucesso!")
    else:
        st.write('---')
        st.warning("Faça o upload de um arquivo .csv para treinar o modelo.")
        st.write('---')


if choice == "Previsão de Conjunto":
    st.write('---')
    st.subheader("Previsão de Conjunto de Dados")
    st.write("Previsão de conjunto de dados via arquivo .csv")
    # selecionei o modelo 'model.pkl'
    if os.path.exists("model.pkl"):
        modelo = pkl.load(open("model.pkl", "rb"))
        # Upload do dataset em .csv
        st.write('---')
        st.subheader("Faça o upload do conjunto de dados para prever")
        file_pred = st.file_uploader("Upload do arquivo CSV", type=["csv"])
        st.write('---')
        if file_pred is not None:
            # leitura do arquivo
            dataframe_pred = pd.read_csv(file_pred, index_col=0)
            # visualização do dataset
            st.subheader("Visualização do conjunto de dados")
            st.write(dataframe_pred)
            st.write('---')
            # botão para prever o conjunto de dados
            st.button("Prever conjunto de dados")
            if st.button:
                # Previ o statos dos alunos
                dataframe_pred['STATUS'] = modelo.predict(dataframe_pred)
                dataframe_pred['STATUS'] = dataframe_pred['STATUS'].apply(lambda x: 'Desistirá' if x == 0 else 'Continuará')
                st.subheader("Visualização do conjunto de dados previsto")
                st.write(dataframe_pred)
                # Download do arquivo .csv
                st.download_button(label="Download do arquivo CSV", data=dataframe_pred.to_csv(), file_name="dataframe_pred.csv", mime="text/csv")
    else:
        st.warning("Faça o treinamento do modelo antes de prever o conjunto de dados.")
st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
