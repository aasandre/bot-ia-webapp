from flask import Flask, render_template, request, jsonify
from threading import Thread
import time
import os
import json
import asyncio

from bot_ia import executar_bot, estatisticas_ia

app = Flask(__name__)

estado_bot = {
    "stake_inicial": 1.0,
    "stake_maxima": 25.0,
    "multiplicador": 2.1,
    "limiar_ia": 0.50,
    "wins": 0,
    "losses": 0,
    "lucro_total": 0.0,
    "ultimos_trades": [],
    "ativo": False
}

def atualizar_stats(result, direcao, stake, lucro):
    estado_bot["wins"] += int(result == "won")
    estado_bot["losses"] += int(result == "lost")
    estado_bot["lucro_total"] += lucro
    estado_bot["ultimos_trades"].append({
        "timestamp": time.strftime("%H:%M:%S"),
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

@app.route("/api/start", methods=["POST"])
def iniciar_bot():
    estado_bot["ativo"] = True
    return jsonify({"status": "bot iniciado"})

@app.route("/api/stop", methods=["POST"])
def parar_bot():
    estado_bot["ativo"] = False
    return jsonify({"status": "bot parado"})

def loop_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        if estado_bot["ativo"]:
            try:
                print("[BOT] Executando ciclo do bot...")
                loop.run_until_complete(executar_bot())
            except Exception as e:
                print("[BOT] Erro durante execução:", str(e))
        else:
            print("[BOT] Aguardando comando para iniciar...")
            time.sleep(5)

if __name__ == "__main__":
    Thread(target=loop_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
