import cv2
import face_recognition
import os
import csv
import streamlit as st
from datetime import datetime
import re
import numpy as np
import time
import pandas as pd

# Configuração do Streamlit
st.set_page_config(page_title="Dashboard de Entrada", layout="wide")

# Título da Aplicação
st.title("Sistema de Entrada - Dashboard")

# Defina o horário de início e término
inicio_horario = "07:30"
fim_horario = "10:15"

# Caminho da pasta onde estão as imagens de referência
pasta_imagens = "C:\\Users\\marcilio\\Desktop\\capturas"
arquivo_csv = "C:\\Users\\marcilio\\Desktop\\Coleta\\registro_entradas.csv"

# Lista para armazenar as codificações de faces e nomes correspondentes
codificacoes_referencia = []
nomes_referencia = []

# Função para limpar e formatar os nomes
def limpar_nome(nome_arquivo):
    nome_limpo = re.sub(r'\d+', '', os.path.splitext(nome_arquivo)[0])
    return nome_limpo

# Carrega todas as imagens da pasta e suas codificações
for arquivo in os.listdir(pasta_imagens):
    if arquivo.endswith(".jpg") or arquivo.endswith(".png"):  # Verifica se é uma imagem
        caminho_imagem = os.path.join(pasta_imagens, arquivo)
        imagem = face_recognition.load_image_file(caminho_imagem)
        try:
            imagem_encoding = face_recognition.face_encodings(imagem)[0]
            codificacoes_referencia.append(imagem_encoding)
            nome = limpar_nome(arquivo)
            nomes_referencia.append(nome)
        except IndexError:
            st.warning(f"Face não encontrada na imagem {arquivo}. Pulando...")

# Substitua o URL RTSP pela conexão com o seu DVR Intelbras
url_rtsp = "rtsp://admin:InTer2022%23@!405@172.16.0.93:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(url_rtsp, cv2.CAP_FFMPEG)

if not cap.isOpened():
    st.error("Erro ao abrir o stream do DVR")
    st.stop()

# Função para verificar se a entrada já foi registrada no dia
def verificar_entrada(nome, data):
    if os.path.exists(arquivo_csv):
        with open(arquivo_csv, 'r') as file:
            registros = csv.reader(file)
            for registro in registros:
                if registro[0] == nome and registro[2] == data:
                    return True
    return False

# Função para registrar a entrada no CSV com verificação
def registrar_entrada(nome, hora):
    data_entrada = datetime.now().strftime("%Y-%m-%d")
    if not verificar_entrada(nome, data_entrada):  # Verifica duplicidade
        with open(arquivo_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([nome, hora, data_entrada])
        st.success(f"Entrada registrada para {nome} às {hora}")
    else:
        st.info(f"Entrada já registrada para {nome} hoje!")

# Função para simular a abertura da porta
def abrir_porta(nome_reconhecido):
    st.info(f"Porta abriu para {nome_reconhecido}...")
    time.sleep(5)  # Simula o tempo que a porta ficaria aberta
    st.info(f"Porta fechou para {nome_reconhecido}...")

# Controle do Streamlit para exibir registros do CSV
st.sidebar.header("Configurações")
mostrar_csv = st.sidebar.checkbox("Mostrar Dados CSV", value=True)

if mostrar_csv:
    st.subheader("Registros de Entradas")
    if os.path.exists(arquivo_csv):
        # Carrega os dados CSV com as colunas atualizadas
        data = pd.read_csv(arquivo_csv, names=["Nome", "Hora de Entrada", "Data de Entrada"])
        st.dataframe(data)  # Exibe a tabela no Streamlit
    else:
        st.warning("Arquivo CSV ainda não foi criado!")

st.sidebar.subheader("Status do Sistema")
st.sidebar.text("Sistema pronto para execução")

# Botão para iniciar o sistema
if st.button("Iniciar Sistema"):
    frame_count = 0  # Contador de frames

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            st.error("Erro ao capturar o frame")
            break

        # Reduz o tamanho do frame para processamento mais rápido (50% do tamanho original)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Obtém o horário atual
        agora = datetime.now().strftime("%H:%M")

        if frame_count % 10 == 0 and inicio_horario <= agora <= fim_horario:
            # Converte o frame para RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(codificacoes_referencia, face_encoding)
                face_distances = face_recognition.face_distance(codificacoes_referencia, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    nome_reconhecido = nomes_referencia[best_match_index]
                    hora_entrada = datetime.now().strftime("%H:%M:%S")
                    registrar_entrada(nome_reconhecido, hora_entrada)
                    abrir_porta(nome_reconhecido)
                else:
                    st.warning("Pessoa desconhecida!")

        time.sleep(0.1)

    cap.release()
    st.sidebar.text("Sistema Parado")
