import asyncio
import websockets
import json
import os
import time
from datetime import datetime

from estado import estado_bot, atualizar_stats

API_TOKEN = os.getenv("DERIV_API_TOKEN", "BZsX6ngxvhEBQOL")
SYMBOL = "1HZ10V"
DURATION = 3

async def send(ws, data, timeout=10, expected_msg_type=None):
    await ws.send(json.dumps(data))
    while True:
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=timeout)
            resp = json.loads(response)
            if "error" in resp:
                print("[ERRO API]", json.dumps(resp, indent=2))
                return None
            if expected_msg_type is None or resp.get("msg_type") == expected_msg_type:
                return resp
        except asyncio.TimeoutError:
            print("[TIMEOUT] Sem resposta da API.")
            return None

async def authorize(ws):
    return await send(ws, {"authorize": API_TOKEN}, expected_msg_type="authorize")

async def subscribe_ticks(ws):
    return await send(ws, {"ticks": SYMBOL, "subscribe": 1})

async def get_proposal(ws, direction):
    contract_type = "PUT" if direction == "RISE" else "CALL"
    return await send(ws, {
        "proposal": 1,
        "amount": estado_bot["stake_inicial"],
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": DURATION,
        "duration_unit": "t",
        "symbol": SYMBOL
    }, expected_msg_type="proposal")

async def buy_contract(ws, proposal_id):
    return await send(ws, {"buy": proposal_id, "price": estado_bot["stake_inicial"]}, expected_msg_type="buy")

async def wait_for_result(ws, contract_id):
    await ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}))
    while True:
        msg = await ws.recv()
        data = json.loads(msg)
        if "proposal_open_contract" in data and data["proposal_open_contract"].get("is_sold"):
            return data["proposal_open_contract"].get("status"), float(data["proposal_open_contract"].get("profit", 0))

async def executar_bot():
    uri = "wss://ws.derivws.com/websockets/v3?app_id=75766"
    async with websockets.connect(uri) as ws:
        if not await authorize(ws):
            print("[ERRO] Falha na autorização.")
            return

        await subscribe_ticks(ws)
        tick_history = []

        while estado_bot["ativo"]:
            msg = await ws.recv()
            data = json.loads(msg)
            if "tick" in data:
                tick = float(data["tick"]["quote"])
                tick_history.append(tick)
                if len(tick_history) > 10:
                    tick_history.pop(0)

                if len(tick_history) < 10:
                    continue

                tick_atual = tick
                media = sum(tick_history) / len(tick_history)
                direction = "RISE" if tick_atual > media else "FALL" if tick_atual < media else None
                if not direction:
                    continue

                print(f"[IA] Direção estimada: {direction}")

                proposal_data = await get_proposal(ws, direction)
                if not proposal_data or "proposal" not in proposal_data:
                    print("[ERRO] Falha ao obter proposta:")
                    print(json.dumps(proposal_data, indent=2))
                    continue

                proposal_id = proposal_data["proposal"]["id"]
                buy_data = await buy_contract(ws, proposal_id)
                if not buy_data or "buy" not in buy_data:
                    print("[ERRO] Falha ao executar compra de contrato:")
                    print(json.dumps(buy_data, indent=2))
                    continue

                contract_id = buy_data["buy"]["contract_id"]
                status, lucro = await wait_for_result(ws, contract_id)

                atualizar_stats(status, direction, estado_bot["stake_inicial"], lucro)
                print(f"[RESULTADO] {status.upper()} | Direção: {direction} | Lucro: {lucro:.2f}")
                await asyncio.sleep(3)

        print("[BOT] Execução encerrada — aguardando nova ativação.")
