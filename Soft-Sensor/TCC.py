import streamlit as st
from PIL import Image
from keras.models import model_from_json
import numpy as np 
import json
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count
import streamlit.components.v1 as components
import time
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Configurando o título

st.set_page_config(page_title=None, page_icon=None, layout='centered', initial_sidebar_state='expanded')

def pag_inicial():

	st.title("Soft Sensor - Redes Neurais Artificiais")
	st.markdown("---")
	st.markdown("### Pedro Oliveira Annoni Farah - Trabalho de Conclusão de Curso  ")
	st.markdown("---")
	st.markdown("Esta aplicação foi desenvolvida para que fosse possível implantar e testar um modelo de predição construído para se estimar o teor de vapor gerado a partir de uma reação de `Combustão` em uma `Caldeira Industrial` de uma siderúrgica real. ")
	st.markdown("---")
	st.markdown("")
	st.markdown("### ⚡ Sobre a proposta")
	st.markdown("")
	#st.write(dict) 

	st.info("A proposta do trabalho foi realizar o desenvolvimento de um Sensor Virtual, isto é, um estimador de uma variável de interesse a partir de variáveis de entrada e que pudesse substituir um sensor físico em uma planta siderúrgica  ")
	st.markdown("### 📊 Sobre os dados")
	st.markdown("")
	st.info("Os dados foram coletados a partir de um banco de dados de Séries Temporais de uma siderúgica real entre os dias 1 e 8 de Agosto de 2022. Eles são oriundos de medidores instalados em uma Caldeira Industrial específica da Central Termoelétrica desta siderúrgica. ")
	st.markdown("### 💻 Sobre o algoritmo")
	st.markdown("")
	st.info("Para o desenvolvimento do Sensor Virtual foi desenvlovido uma Rede Neural Aritificial capaz de predizer um valor de saída de interesse, no caso, a quantidade de vapor gerada a partir das variaáveis de entrada coletadas. ")

	st.markdown("<div align='center'><br>"
                "<img src='https://img.shields.io/badge/Feito%20COM-PYTHON%20-red?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/EDITADO%20COM-SUBLIME-blue?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/DESENVOLVIDO%20COM-Streamlit-green?style=for-the-badge'"
                "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
	
	st.markdown("---")
add_sidebar = st.sidebar.selectbox("Opções de escolha: ", ('Início','Base','Previsão e testes'))
if(add_sidebar == 'Início'):
	pag_inicial()





if (add_sidebar == 'Base'):
	st.title("📑 Base")
	st.markdown("---")
	st.markdown("As variáveis de entrada utilizadas para previsão da variável de saída foram escolhidas com base em estudos relacionados ao processo de Combustão como um todo. Sabe-se que dois dos principais componentes de uma reação de Combustão são os `Combustíveis` e os `Comburentes`. O primeiro pode ser composto por diversas substâncias, podendo inclusive ser sólido, líquido ou gasoso. Enquanto isso, o comburente geralmente é o próprio ar atmosférico. Para o caso estudado, as variáveis de entrada utilizadss nessa caldeira são as seguintes: ")
	'''
	- [x]  Gás de Alto Forno - GAF (Nm3/h) 
	- [x]  Gás de Coqueria - GCO (Nm3/h)
	- [x]  Gás Natural - GN (Nm3/h)
	- [x]  Gás Nitrogênio - N2 (Nm3/h)
	'''
	st.markdown("")
	st.warning("Juntamente com os combustíveis que alimentam a Caldeira está a quantidade de Ar fornecido como comburente para a reação. Caso a quantidade seja insuficiente, tem-se a Combustão Incompleta, onde nem toda a quantidade de combustível reage, levando à formação do tóxico Monóxido de Carbono. Caso a quantidade esteja em excesso, apesar de reagir com todo o combustível, esta quantidade adicional acaba por roubar calor dos produtos diminuindo a eficiência da reação.")
	st.markdown("")
	'''
	Abaixo é possível escolher uma das variáveis de entrada para visualizar o seu comportamento durante os dias 1 e 8 de agosto de 2022: 
	'''
	option = st.selectbox("Qual variável gostaria de visualizar?",("GAF","GCO","GN","N2")) 
	st.markdown("")
	if (option == "GAF"):
		image = Image.open('GAFIMG.png')
		st.image(image, caption='Gás de Alto Forno (Nm3/h)')
	if (option == "GCO"):
		image = Image.open('GCOIMG.png')
		st.image(image, caption='Gás de Coqueria (Nm3/h)')
	if (option == "GN"):
		image = Image.open('GNIMG.png')
		st.image(image, caption='Gás Natural (Nm3/h)')
	if (option == "N2"):
		image = Image.open('N2IMG.png')
		st.image(image, caption='Gás Nitrogênio (Nm3/h)')
    
	st.markdown("Um parâmetro interessante que pode ser visualizado é a correlação entre as variáveis de entrada e saída. Neste caso, a variável de saída está sendo considerada como a quantidade de vapor, mas ao fim será o teor de Oxigênio: ")
	image = Image.open('CORR.png')
	st.image(image, caption='Correlação entre variáveis')
	st.markdown("")
	st.markdown("Um parâmetro interessante que pode ser visualizado é a correlação entre as variáveis de entrada e saída. Neste caso, a variável de saída está sendo considerada como a quantidade de vapor, mas ao fim será o teor de Oxigênio: ")
	st.markdown("")
	st.markdown("A princípio, a Rede Neural foi montada com os seguintes parâmetros: ")
	'''
	- [x]  2 camadas ocultas 
	- [x]  3 neurônios em cada camada oculta
	- [x]  Função de Ativação 'reLU' entre as camadas ocultas
	- [x]  Otimizador 'Adam'
	- [x]  Função de Perda 'mean absolute error'
	- [x]  Métrica principal também 'mean absolute error'
	'''
	st.markdown("")
	st.markdown("Para um treinamento com 30 épocas e tamanho do Lote igual a 50, obteve-se os valores que podem ser comparados com os valores reais no gráfico abaixo:  ")
	image = Image.open('RESCOMP.png')
	st.image(image, caption='Comparação entre valores previstos e reais')



if (add_sidebar == 'Previsão e testes'):
	st.title("🔍 Previsão e testes!")
	st.markdown("---")
	'''
	O funcionamento do teste é bastante simples e intuitivo. É necessário que você informe os valoers para as seguintes variáveis:
 1. Fluxo de Gás de Alto Forno
 2. Fluxo de Gás de Coqueria
 3. Fluxo de Gás Natural 
 4. Fluxo de Gás Nitrogênio
	'''
	st.markdown("---")
	st.markdown("### Gás de Alto Forno")
	st.markdown("Primeiramente, é necessário fornecer a vazão de Gás de Alto Forno que estará alimentando a Caldeira Industrial nesse momento: ")
	GAF = st.number_input('Vazão de Gás de Alto Forno (Nm3/h)')
	st.markdown("### Gás de Coqueria")
	st.markdown("Do mesmo modo, também é necessário informar a vazão de Gás de Coqueria: ")
	GCO = st.number_input('Vazão de Gás de Coqueria (Nm3/h)')
	st.markdown("### Gás Natural")
	st.markdown("Escolha agora a vazão de Gás Natural através do slider:  ")
	GN = st.slider('Vazão de Gás Natural (Nm3/h)', 0,15000)
	st.markdown("### Gás Nitrogênio")
	st.markdown("Agora, escolher a vazão de Gás Nitrogênio através do slider:  ")
	N2 = st.slider('Vazão de Gás Nitrogênio (Nm3/h)', 0,15000)

	previsor = np.array([[GAF,GCO,GN,N2]])
	if st.button("Clique aqui para calcular a quantidade vapor esperada!"):
		arquivo = open('./Regressorr.json', 'r')
		estrutura = arquivo.read()
		arquivo.close()

		regressor = model_from_json(estrutura)
		regressor.load_weights('./Regressorr.h5')
		resultado = regressor.predict(previsor)
		st.info(resultado[0][0])

	carregar = st.selectbox("Carregar dados para prever saídas?: ", ('Sim','Não'))
	if carregar == "Sim":
		upload_file = st.file_uploader("Escolha um arquivo CSV", type = 'csv')
		if upload_file is not None:
			data = pd.read_csv(upload_file)
			st.write(data)
			st.success("Dados importados com sucesso")
			if st.button("Estimar valores..."):
				## Tirar isso depois
				resultados = []
				x_vals = []
				y_vals = []
				y_vals2 = []
				previsores = data.iloc[0:1440,1:5].values
				valorReal = data.iloc[0:1440,5].values;
				arquivo = open('./Regressorr.json', 'r')
				estrutura = arquivo.read()
				arquivo.close()
				regressor = model_from_json(estrutura)
				regressor.load_weights('./Regressorr.h5')
				previsoes = regressor.predict(previsores)
				previsoes = pd.DataFrame(previsoes, columns = ['Previsão'])
				valorReal = pd.DataFrame(valorReal, columns = ['Vapor'])
				time1 = data.iloc[0:1440,0].values
				time1 = pd.DataFrame(time1, columns = ['Data'])
				df  = pd.concat([time1,previsoes], axis=1)
				df2 = pd.concat([time1,valorReal], axis=1)
				x = df['Data'].values
				y = df['Previsão'].values
				y2 = df2['Vapor'].values
				index = count()
				plt.style.use('fivethirtyeight')
				##the_plot = st.pyplot(plt)
				fig = px.line()
				##fig2 = px.line()
				
				subfig = make_subplots(specs=[[{"secondary_y":True}]])
				the_plot = st.plotly_chart(subfig)
				def animate (i):
					x_vals.append(x[i])
					y_vals.append(y[i])
					y_vals2.append(y2[i])

					plt.cla()
					plt.plot(x_vals,y_vals)
					#plt.plot(x_vals,y_vals2)
					plt.title("Valores estimados ao longo do tempo")
					plt.xlabel("Tempo")
					plt.ylabel("Quantidade de Vapor")
					if i == 0:
						subfig.add_trace(go.Scatter(x = x_vals, y = y_vals, name = "Previstos",marker= {'color': '#636EFA'}))
						subfig.add_trace(go.Scatter(x = x_vals, y = y_vals2, name = "Reais",marker= {'color': 'orange'}))
					else:
						subfig.add_trace(go.Scatter(x = x_vals, y = y_vals, showlegend = False, marker= {'color': '#636EFA'}))
						subfig.add_trace(go.Scatter(x = x_vals, y = y_vals2, showlegend = False,marker= {'color': 'orange'}))
					subfig.update_xaxes(title_text = "Tempo (minuto)")
					
					subfig.update_yaxes(title_text = "Quantidade de Vapor")
					#subfig.update_yaxes(title_text = "Tempo (minuto)")
					#fig = px.line(x = x_vals, y = y_vals, title = "Previsões ao longo do tempo")
					#fig2 = px.line(x = x_vals, y = y_vals2)
					#fig2.update_traces(yaxis = "y2")
					
					#subfig.add_traces(fig.data + fig2.data)
					#subfig.layout.xaxis.title = "Tempo (minuto)"
					#subfig.layout.yaxis.title = "Quantidade de Vapor"
					#subfig.layout.xaxis.title = "Tempo (minuto)"
					#subfig.layout.yaxis.title = "Quantidade de Vapor"
					subfig.update_traces(mode='markers+lines')
					subfig.update_layout(
    					xaxis_title="Tempo (minuto)",
    					yaxis_title="Quantidade de Vapor",
					yaxis_range=[60,80]
    					)
					#subfig.for_each_trace(lambda t: t.update(line = dict(color = t.marker.color)))

					#fig.show()
					#plt.tight_layout()
					#the_plot.pyplot(plt)
					the_plot.plotly_chart(subfig)
					#the_plot2.plotly_chart(fig2)
				for i in range(1000):
					animate(i)
					time.sleep(10)
				#ani = FuncAnimation(plt.gcf(), animate, interval = 10000)
