import pandas as pd
from transformers import pipeline

# Carregar o pipeline de análise de sentimentos em português
sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Carregar o arquivo Excel
arquivo_excel = "comentario.xlsx"
df = pd.read_excel(arquivo_excel, engine="openpyxl")

# Nome das colunas
coluna_comentario = "comentario"
coluna_sentimento = "sentimento"

# Verifica se a coluna existe
if coluna_comentario in df.columns:
    def analisar_comentario(texto):
        if isinstance(texto, str) and texto.strip():
            resultado = sentiment_pipeline(texto)
            label = resultado[0]['label']

            # Converter saída do modelo para uma classificação mais simples
            if "1 star" in label or "2 stars" in label:
                return "negativo"
            elif "4 stars" in label or "5 stars" in label:
                return "positivo"
            else:
                return "neutro"
        return "neutro"

    # Aplicar a análise de sentimento
    df[coluna_sentimento] = df[coluna_comentario].astype(str).apply(analisar_comentario)

    # Salvar o resultado
    df.to_excel("comentarios_analisados.xlsx", index=False, engine="openpyxl")
    print("✅ Análise concluída com BERTimbau!")

else:
    print(f"⚠️ A coluna '{coluna_comentario}' não foi encontrada na planilha.")
