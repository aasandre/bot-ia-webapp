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
    from datetime import datetime
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
