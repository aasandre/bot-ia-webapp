from flask import Flask, render_template, request, jsonify
from threading import Thread
import time
import os
import json

# Importa o bot e suas variáveis principais
import asyncio
import websockets
import torch
from datetime import datetime

from Bot_IA_PyTorch_V10_1s_v12_Final_Reconectavel_Backup import executar_bot, estatisticas_ia

app = Flask(__name__)

# Estado compartilhado
estado_bot = {
    "stake_inicial": 1.0,
    "stake_maxima": 25.0,
    "multiplicador": 2.1,
    "limiar_ia": 0.50,
    "wins": 0,
    "losses": 0,
    "lucro_total": 0.0,
    "ultimos_trades": []
}

# Atualizar estatísticas do bot
def atualizar_stats(result, direcao, stake, lucro):
    estado_bot["wins"] += int(result == "won")
    estado_bot["losses"] += int(result == "lost")
    estado_bot["lucro_total"] += lucro
    estado_bot["ultimos_trades"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "direcao": direcao,
        "stake": stake,
        "resultado": result,
        "lucro": round(lucro, 2)
    })
    if len(estado_bot["ultimos_trades"]) > 20:
        estado_bot["ultimos_trades"].pop(0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/estatisticas")
def get_stats():
    return jsonify(estado_bot)

@app.route("/api/atualizar_parametros", methods=["POST"])
def atualizar_parametros():
    dados = request.json
    estado_bot["stake_inicial"] = float(dados.get("stake_inicial", 1))
    estado_bot["stake_maxima"] = float(dados.get("stake_maxima", 25))
    estado_bot["multiplicador"] = float(dados.get("multiplicador", 2.1))
    estado_bot["limiar_ia"] = float(dados.get("limiar_ia", 0.50))
    return jsonify({"status": "ok"})

@app.route("/api/reset", methods=["POST"])
def resetar_stats():
    estado_bot["wins"] = 0
    estado_bot["losses"] = 0
    estado_bot["lucro_total"] = 0.0
    estado_bot["ultimos_trades"].clear()
    return jsonify({"status": "resetado"})

def rodar_bot_em_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(executar_bot())
    except Exception as e:
        print("[ERRO NO BOT]", e)

if __name__ == "__main__":
    # Iniciar bot em thread separada
    Thread(target=rodar_bot_em_thread, daemon=True).start()
    # Rodar servidor Flask
    app.run(debug=False)
