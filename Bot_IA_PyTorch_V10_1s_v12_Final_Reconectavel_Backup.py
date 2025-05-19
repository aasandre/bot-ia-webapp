
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import os
from datetime import datetime
from collections import deque
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import os

def salvar_modelo_ensemble(ia, path_model="modelo_ensemble.pt", path_scaler="scaler_ensemble.pkl"):
    torch.save(ia.modelos[0].modelo.state_dict(), path_model)
    joblib.dump(ia.modelos[0].scaler, path_scaler)

def carregar_modelo_ensemble(input_cols, path_model="modelo_ensemble.pt", path_scaler="scaler_ensemble.pkl"):
    ia = EnsembleIA_PyTorch(input_cols)
    for modelo in ia.modelos:
        modelo.modelo.load_state_dict(torch.load(path_model))
        modelo.scaler = joblib.load(path_scaler)
    return ia

def salvar_modelo_contexto(ia_contexto, path_model="modelo_contexto.pt", path_scaler="scaler_contexto.pkl"):
    torch.save(ia_contexto.modelo.state_dict(), path_model)
    joblib.dump(ia_contexto.scaler, path_scaler)

def carregar_modelo_contexto(input_cols, path_model="modelo_contexto.pt", path_scaler="scaler_contexto.pkl"):
    ia = IA_Contexto_PyTorch(input_cols)
    ia.modelo.load_state_dict(torch.load(path_model))
    ia.scaler = joblib.load(path_scaler)
    return ia


# Constantes
API_TOKEN = "BZsX6ngxvhEBQOL"
SYMBOL = "1HZ10V"
DURATION = 3
STAKE_INICIAL = 1
MAX_PERDAS = 3
LIMIAR_DADOS_IA = 100
CAMINHO_CSV = "dados_treinamento.csv"
DIAGNOSTICO_CSV = "diagnostico_ia.csv"

# Variáveis globais
stake_atual = STAKE_INICIAL
perdas_consecutivas = 0
modo_fake = False
tick_history = deque(maxlen=100)
wins = 0
losses = 0
lucro_total = 0.0
features_usadas = []

estatisticas_ia = {
    "acertos": 0,
    "erros": 0,
    "bloqueios": 0,
    "total": 0,
    "limiar_conf": 0.50
}

# Modelo MLP em PyTorch
class ModeloMLPPyTorch(nn.Module):
    def __init__(self, input_size):
        super(ModeloMLPPyTorch, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.rede(x)

# Classe da IA
class IA_PyTorch:
    def __init__(self, input_cols):
        self.input_cols = input_cols
        self.modelo = ModeloMLPPyTorch(len(input_cols))
        self.scaler = StandardScaler()
        self.optim = torch.optim.Adam(self.modelo.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def treinar(self, df):
        from sklearn.utils import shuffle
        if len(self.input_cols) == 0:
            print("[ERRO] Nenhuma feature definida — treino abortado.")
            return

        df = shuffle(df)
        X = df[self.input_cols].fillna(0).values
        y = df['resultado'].map({'won': 1, 'lost': 0}).values

        # Validações de segurança
        if np.isnan(X).any():
            print("[ERRO] Dados de entrada contêm NaNs — treino abortado.")
            return
        if len(X) < 10:
            print("[ERRO] Dataset muito pequeno para treino.")
            return
        if len(set(y)) < 2:
            print("[ERRO] Labels únicos — não há variação no target.")
            return

        # Penalização por streaks discretos
        streaks = df.get("streak_lost", pd.Series([0]*len(df))).values
        pesos_streak = np.select(
            [streaks == 1, streaks == 2, streaks >= 3],
            [1.3, 2.0, 3.0],
            default=1.0
        )

        # Penalização por baixa entropia (suavizada)
        entropia = np.std(X, axis=1)
        pesos_entropia = np.clip(1.0 + (1.0 / (entropia + 1e-4)), 1.0, 10.0)

        # Peso final
        pesos = pesos_streak * pesos_entropia

        # Treino normal
        X_scaled = self.scaler.fit_transform(X)
        if np.isnan(X_scaled).any():
            print("[ERRO] Scaler retornou NaN — verifique os dados.")
            return

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

        melhor_loss = float('inf')
        paciencia = 5
        epocas_sem_melhora = 0

        self.modelo.train()
        for epoca in range(30):
            self.optim.zero_grad()
            saida = self.modelo(X_tensor)
            losses = self.loss_fn(saida, y_tensor)
            loss = (losses * pesos_tensor).mean()
            if torch.isnan(loss):
                print("[ERRO] Loss virou NaN — treino cancelado.")
                return
            loss.backward()
            self.optim.step()

            if loss.item() < melhor_loss - 0.0001:
                melhor_loss = loss.item()
                epocas_sem_melhora = 0
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= paciencia:
                    print(f"[EARLY STOP] Interrompido na época {epoca} (loss: {loss.item():.5f})")
                    break
    def prever(self, linha):
        if not hasattr(self.scaler, "mean_"):
            print("[ERRO] Scaler ainda não treinado.")
            return "lost", 0.0

        self.modelo.eval()
        for col in self.input_cols:
            if col not in linha.columns:
                linha[col] = 0
        X = linha[self.input_cols].fillna(0).values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        if np.isnan(X_scaled).any():
            print("[ERRO] Scaler retornou NaN na previsão.")
            return "lost", 0.0

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            saida = self.modelo(X_tensor)
            probs = F.softmax(saida, dim=1).numpy()[0]
            classe = 'won' if probs[1] >= probs[0] else 'lost'
            confianca = max(probs)
            return classe, confianca

class EnsembleIA_PyTorch:
    def __init__(self, input_cols, n_modelos=3):
        self.modelos = [IA_PyTorch(input_cols) for _ in range(n_modelos)]
        self.input_cols = input_cols
        self.n_modelos = n_modelos

    def treinar(self, df):
        for modelo in self.modelos:
            modelo.treinar(df)

    def prever(self, linha):
        votos = []
        confiancas = []
        for modelo in self.modelos:
            try:
                classe, confianca = modelo.prever(linha)
                votos.append(classe)
                confiancas.append(confianca)
            except Exception as e:
                print(f"[ERRO ENSEMBLE] Falha na previsão: {e}")
                votos.append("lost")
                confiancas.append(0.0)

        classe_final = max(set(votos), key=votos.count)
        media_confianca = np.mean(confiancas)
        return classe_final, media_confianca


# IA de contexto
class IA_Contexto_PyTorch:
    def __init__(self, input_cols):
        self.input_cols = input_cols
        self.modelo = ModeloMLPPyTorch(len(input_cols))
        self.scaler = StandardScaler()
        self.optim = torch.optim.Adam(self.modelo.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def treinar(self, df):
        df = shuffle(df)
        X = df[self.input_cols].fillna(0).values
        y = df['contexto'].map({'seguro': 1, 'arriscado': 0}).values
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        self.modelo.train()
        for _ in range(30):
            self.optim.zero_grad()
            out = self.modelo(X_tensor)
            loss = self.loss_fn(out, y_tensor)
            loss.mean().backward()
            self.optim.step()

        self.modelo.eval()
        with torch.no_grad():
            preds = self.modelo(X_tensor).argmax(dim=1)
            acc = (preds == y_tensor).float().mean().item()
            
    def prever(self, linha):
        if not hasattr(self.scaler, "mean_"):
            print("[ERRO] Scaler ainda não treinado.")
            return "arriscado", 0.0
    
        self.modelo.eval()
        for col in self.input_cols:
            if col not in linha.columns:
                linha[col] = 0
        X = linha[self.input_cols].fillna(0).values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
    
        if np.isnan(X_scaled).any():
            print("[ERRO] Scaler retornou NaN na previsão.")
            return "arriscado", 0.0

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            saida = self.modelo(X_tensor)
            probs = F.softmax(saida, dim=1).numpy()[0]
            classe = 'seguro' if probs[1] >= probs[0] else 'arriscado'
            confianca = max(probs)
            return classe, confianca

# IAs serão instanciadas dinamicamente
ia = None
ia_contexto = None
features_contexto = []

# Funções auxiliares
def extrair_indicadores(ticks):
    if len(ticks) < 10:
        return None
    df = pd.DataFrame(ticks, columns=["close"])
    df["high"] = df["close"] + 0.01
    df["low"] = df["close"] - 0.01

    # Bandas de Bollinger
    from ta.volatility import BollingerBands, AverageTrueRange
    bb = BollingerBands(df["close"])
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()

    # Volatilidade
    df["var"] = df["close"].pct_change().rolling(window=5).std()

    # EMA curtas
    df["ema_3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()

    # Delta de preço
    df["delta_1"] = df["close"].diff(1)
    df["delta_2"] = df["close"].diff(2)

    # Retorno acumulado em 3 ticks
    df["ret_3"] = df["close"].pct_change(periods=3)

    
    # ATR curta
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=3)
    df["atr_3"] = atr.average_true_range()

    return df.iloc[-1].dropna()

def carregar_dados():
    return pd.read_csv(CAMINHO_CSV) if os.path.exists(CAMINHO_CSV) else pd.DataFrame()

def registrar_dados(indicadores, resultado):
    global ia, ia_contexto, features_usadas, features_contexto

    dados = carregar_dados()
    nova_linha = indicadores.to_frame().T
    nova_linha["resultado"] = resultado

    # Streak de perdas consecutivas
    streak_lost = 0
    if "resultado" in dados.columns:
        for res in reversed(dados["resultado"].tolist()):
            if res == "lost":
                streak_lost += 1
            else:
                break
    if resultado == "lost":
        streak_lost += 1
    nova_linha["streak_lost"] = streak_lost

    # Classificar como arriscado ou seguro
    contexto = "seguro"
    if "resultado" in dados.columns:
        ultimos = dados["resultado"].tolist()[-3:] + [resultado]
        if ultimos[-3:] == ["lost", "lost", "lost"]:
            contexto = "arriscado"
        elif resultado == "won":
            contexto = "seguro"

    nova_linha["contexto"] = contexto

    dados = pd.concat([dados, nova_linha], ignore_index=True)

    # SISTEMA DE ESQUECIMENTO INTELIGENTE
    if dados["resultado"].nunique() == 2 and len(dados) > 5000:
        dados_recent = dados.tail(3000)
        dados_lost = dados[dados["resultado"] == "lost"].tail(1000)
        dados_info = dados.sort_values(by="var", ascending=False).drop_duplicates().head(1000)
        dados = pd.concat([dados_recent, dados_lost, dados_info])
        dados = dados.drop_duplicates().sample(frac=1).tail(5000)  # embaralhar e truncar

    dados.to_csv(CAMINHO_CSV, index=False)

    # TREINAMENTO
    if ia is None and "resultado" in dados.columns and len(dados["resultado"]) >= LIMIAR_DADOS_IA and len(features_usadas) == 0:
        features_usadas = list(dados.columns.drop(["resultado", "contexto"]))
    
    if os.path.exists("modelo_ensemble.pt") and os.path.exists("scaler_ensemble.pkl") and len(features_usadas) > 0:
        ia = carregar_modelo_ensemble(features_usadas)
    else:
        print("[TREINO] Nenhum modelo salvo encontrado. Treinando IA inicial...")
        ia = EnsembleIA_PyTorch(features_usadas)
        ia.treinar(dados)
        salvar_modelo_ensemble(ia)
        globals()["ia"] = ia

    if os.path.exists("modelo_contexto.pt") and os.path.exists("scaler_contexto.pkl") and len(features_contexto) > 0:
        ia_contexto = carregar_modelo_contexto(features_contexto)

    else:
            print("[TREINO] Nenhum modelo salvo encontrado. Treinando IA inicial...")
            ia = EnsembleIA_PyTorch(features_usadas)
            ia.treinar(dados)
            salvar_modelo_ensemble(ia)
            globals()["ia"] = ia

    if ia is not None:
        ia.treinar(dados)

    if "contexto" in dados.columns:
        features_contexto = list(dados.columns.drop(["resultado", "contexto"]))
        if "streak_lost" not in features_contexto:
            features_contexto.append("streak_lost")
        if ia_contexto is None and len(features_contexto) > 0:
            ia_contexto = IA_Contexto_PyTorch(features_contexto)
        ia_contexto.treinar(dados)
        salvar_modelo_contexto(ia_contexto)
        globals()["ia_contexto"] = ia_contexto


async def simular_operacao(direction, entrada_tick, ws):
    print("[SIMULACAO] IA bloqueou a entrada. Simulando operacao...")

    ticks_recebidos = []
    await ws.send(json.dumps({"forget_all": "ticks"}))
    await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

    # Captura o primeiro tick como ponto de entrada
    entrada_tick_valido = None
    while entrada_tick_valido is None:
        msg = await ws.recv()
        data = json.loads(msg)
        if "tick" in data and data["tick"]["symbol"] == SYMBOL:
            entrada_tick_valido = float(data["tick"]["quote"])
            ticks_recebidos.append(entrada_tick_valido)

    # Coleta exatamente 2 novos ticks distintos (simulação ajustada para 3 ticks reais)
    ultimo_tick = entrada_tick_valido
    while len(ticks_recebidos) < 3:
        msg = await ws.recv()
        data = json.loads(msg)
        if "tick" in data and data["tick"]["symbol"] == SYMBOL:
            preco = float(data["tick"]["quote"])
            if preco != ultimo_tick:
                ticks_recebidos.append(preco)
                ultimo_tick = preco

    saida = ticks_recebidos[-1]
    resultado = "won" if (direction == "RISE" and saida >= entrada_tick_valido) or \
                         (direction == "FALL" and saida <= entrada_tick_valido) else "lost"

    print(f"[SIMULACAO] Resultado simulado: {resultado} (entrada: {entrada_tick_valido}, saida: {saida})")
    return resultado

async def send(ws, data, timeout=10):
    await ws.send(json.dumps(data))
    try:
        response = await asyncio.wait_for(ws.recv(), timeout=timeout)
        resp = json.loads(response)
        if "error" in resp and resp.get("msg_type") == "buy":
            print("[ERRO API]", json.dumps(resp, indent=2))
            return None
        return resp
    except:
        return None

async def authorize(ws):
    return await send(ws, {"authorize": API_TOKEN})

async def subscribe_ticks(ws):
    return await send(ws, {"ticks": SYMBOL, "subscribe": 1})

async def get_proposal(ws, direction):
    contract_type = "PUT" if direction == "RISE" else "CALL"
    return await send(ws, {
        "proposal": 1,
        "amount": stake_atual,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": DURATION,
        "duration_unit": "t",
        "symbol": SYMBOL
    })

async def buy_contract(ws, proposal_id):
    return await send(ws, {"buy": proposal_id, "price": stake_atual})

async def wait_for_result(ws, contract_id):
    await ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}))
    while True:
        msg = await ws.recv()
        data = json.loads(msg)
        if "proposal_open_contract" in data and data["proposal_open_contract"].get("is_sold"):
            return data["proposal_open_contract"].get("status"), float(data["proposal_open_contract"].get("profit", 0))

async def executar_bot():
    global stake_atual, perdas_consecutivas, wins, losses, lucro_total, ia_contexto
    if not os.path.exists(DIAGNOSTICO_CSV):
        with open(DIAGNOSTICO_CSV, "w") as f:
            f.write("timestamp,confiança,previsão,resultado_real,decisao,limiar,acertos,erros,bloqueios")

    uri = "wss://ws.derivws.com/websockets/v3?app_id=75766"
    async with websockets.connect(uri) as ws:
        if not await authorize(ws): return
        await subscribe_ticks(ws)
        dados = carregar_dados()
        features_contexto = [
            'bb_delta', 'ema3_delta', 'ema5_delta',
            'delta_1', 'delta_2', 'retorno_3',
            'atr', 'equals_1', 'equals_2', 'variancia',
            'streak_lost', 'preco_normalizado'
        ]

        if len(dados) >= LIMIAR_DADOS_IA and "resultado" in dados.columns:
            global features_usadas, ia
            features_usadas = list(dados.columns.drop(["resultado", "contexto"]))
            ia = EnsembleIA_PyTorch(features_usadas)
            ia.treinar(dados)
        if "contexto" in dados.columns:
            features_contexto = list(dados.columns.drop(["resultado", "contexto"]))
            ia_contexto = IA_Contexto_PyTorch(features_contexto)
            ia_contexto.treinar(dados)

        async for msg in ws:
            predito = 'unknown'
            confianca = 0.0
            data = json.loads(msg)
            if "tick" in data:
                tick = float(data["tick"]["quote"])
                tick_history.append(tick)

                if len(tick_history) < 30:
                    continue

                indicadores = extrair_indicadores(list(tick_history))
                if indicadores is None or indicadores.isnull().any():
                    continue

                tick_atual = tick
                ema = np.mean(list(tick_history)[-10:])
                direction = "RISE" if tick_atual > ema else "FALL" if tick_atual < ema else None
                if not direction:
                    continue

                permitir = True
                if ia:
                    predito, confianca = ia.prever(pd.DataFrame([indicadores]))
                    permitir = confianca >= estatisticas_ia["limiar_conf"] and predito == "won"
                    if ia_contexto:
                        contexto_predito, conf_ctx = ia_contexto.prever(pd.DataFrame([indicadores]))
                        permitir = permitir and (contexto_predito == "seguro")
                        if not permitir:
                            print("[IA BLOQUEIO] Entrada bloqueada por contexto ou baixa confianca.")
                        print(f"[IA CONTEXTO] Previsao: {contexto_predito} | Entrada sera {'liberada' if permitir else 'bloqueada'}")
                        estado = "liberada" if permitir else "bloqueada"
                        print(f"[IA] Confianca: {confianca:.2f} | Previsao: {predito} | Entrada sera {estado}.")
                else:
                    resultado_simulado = await simular_operacao(direction, tick_atual, ws)
                    registrar_dados(indicadores, resultado_simulado)
                    continue
                    print("[IA] Sem modelo ativo no momento.")

                if permitir:                    
                    proposal_data = await get_proposal(ws, direction)
                    if not proposal_data or "proposal" not in proposal_data:
                        print("[ERRO] Falha ao obter proposta de contrato:")
                        print(json.dumps(proposal_data, indent=2))
                        continue
                    if not proposal_data or "proposal" not in proposal_data: continue
                    proposal_id = proposal_data["proposal"]["id"]
                    buy_data = await buy_contract(ws, proposal_id)
                    if not buy_data or "buy" not in buy_data:
                        print("[ERRO] Falha ao executar compra de contrato:")
                        print(json.dumps(buy_data, indent=2))
                        continue
                    if not buy_data or "buy" not in buy_data: continue
                    contract_id = buy_data["buy"]["contract_id"]
                    status, lucro = await wait_for_result(ws, contract_id)
                    resultado = "won" if status == "won" else "lost"
                    


                    if resultado == "won":
                        perdas_consecutivas = 0
                        wins += 1
                        stake_atual = STAKE_INICIAL
                    else:
                        perdas_consecutivas += 1
                        stake_atual = round(min(stake_atual * 2.1, 25), 2)
                        losses += 1
                    lucro_total += lucro

                    print(f"[RESULTADO] Resultado: {resultado} | Direcao: {direction} | Lucro: {lucro:.2f} | Wins: {wins} | Losses: {losses} | Lucro Total: {lucro_total:.2f}")

                    estatisticas_ia["total"] += 1
                    if ia and confianca >= estatisticas_ia["limiar_conf"]:
                        if predito == resultado:
                            estatisticas_ia["acertos"] += 1
                        else:
                            estatisticas_ia["erros"] += 1

                    with open(DIAGNOSTICO_CSV, "a") as diag:
                        diag.write(f"{datetime.now()},{confianca:.2f},{predito},{resultado},REAL,{estatisticas_ia['limiar_conf']:.2f},{estatisticas_ia['acertos']},{estatisticas_ia['erros']},{estatisticas_ia['bloqueios']}\n")

                    registrar_dados(indicadores, resultado)
                else:
                    estatisticas_ia["bloqueios"] += 1
                    confianca = 0.0
                    predito = 'lost'
                    resultado_simulado = await simular_operacao(direction, tick_atual, ws)
                    with open(DIAGNOSTICO_CSV, "a") as diag:
                        diag.write(f"{datetime.now()},{confianca:.2f},{predito},{resultado_simulado},SIMULADO,{estatisticas_ia['limiar_conf']:.2f},{estatisticas_ia['acertos']},{estatisticas_ia['erros']},{estatisticas_ia['bloqueios']}\n")
                    registrar_dados(indicadores, resultado_simulado)

import time
import traceback

if __name__ == "__main__":
    tentativas = 0
    while True:
        try:
            print("[BOT] Iniciando execucao...")
            asyncio.run(executar_bot())
        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.InvalidStatusCode,
                ConnectionResetError,
                asyncio.TimeoutError) as e:
            tentativas += 1
            tempo_espera = min(60, tentativas * 5)  # espera crescente até 60s
            print(f"[RECONEXAO] Erro de conexao detectado: {str(e)}")
            print(f"[RECONEXAO] Tentando reconectar em {tempo_espera} segundos... (tentativa {tentativas})")
            time.sleep(tempo_espera)
        except Exception as e:
            print("[ERRO FATAL] Ocorreu um erro nao previsto:")
            traceback.print_exc()
            break
