import asyncio
import json
import websockets

async def test():
    print("[*] Connecting to local WebSocket: ws://127.0.0.1:8000/ws/deribit ...")
    async with websockets.connect("ws://127.0.0.1:8000/ws/deribit") as ws:
        print("[✓] Connected! Streaming 3 live ticks...")
        for i in range(3):
            msg = await ws.recv()
            data = json.loads(msg)
            m = data["live_market"]
            ai = data["ai_prediction"]["primary_prediction"]
            print(f"  Tick #{i+1}: BTC ${m['mid_price']:,.2f} | Spread: ${m['bid_ask_spread']:.2f} | Regime: {ai['liquidity_regime']} | Crash Risk (60s): {ai['flash_crash_probability_60s']*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(test())
