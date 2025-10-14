from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/queimadas/ano/{ano}") # 
def focos_por_ano(ano: int):
    try:
        # Encontra todos os arquivos CSV na pasta 'data/'
        arquivos_csv = glob.glob(os.path.join("data", "*.csv"))

        if not arquivos_csv:
            return JSONResponse(content={"erro": "Nenhum arquivo CSV encontrado"}, status_code=404)

        # Lista para armazenar os DataFrames
        dataframes = []

        for arquivo in arquivos_csv:
            df = pd.read_csv(arquivo, encoding="latin1", sep=",")
            if "DataHora" in df.columns:
                df["DataHora"] = pd.to_datetime(df["DataHora"], errors="coerce")
                df = df[df["DataHora"].dt.year == ano]
                dataframes.append(df)

        # Se nenhum dado do ano foi encontrado
        if not dataframes:
            return JSONResponse(content={"erro": f"Nenhum dado encontrado para o ano {ano}"}, status_code=404)

        # Concatena todos os DataFrames
        df_total = pd.concat(dataframes, ignore_index=True)

        # Agrupa os dados por data e conta os focos
        df_total["data"] = df_total["DataHora"].dt.date
        grupo = df_total.groupby("data").size().reset_index(name="qtd_focos")

        return grupo.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
