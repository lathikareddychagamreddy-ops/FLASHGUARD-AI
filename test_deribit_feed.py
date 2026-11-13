"""
Deribit WebSocket API v2 Live Feed Verifier
Tests connection to wss://www.deribit.com/ws/api/v2,
subscribes to BTC-PERPETUAL ticker and orderbook channels,
and receives live tick data.
"""

import asyncio
import json
import websockets
import sys

DERIBIT_WS_URL = "wss://www.deribit.com/ws/api/v2"

async def test_deribit_feed():
    print(f"[*] Connecting to Deribit WebSocket: {DERIBIT_WS_URL} ...")
    try:
        async with websockets.connect(DERIBIT_WS_URL) as ws:
            print("[✓] Connected successfully to Deribit WebSocket v2.")

            # 1. Test ping / test method
            ping_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/test",
                "params": {}
            }
            await ws.send(json.dumps(ping_msg))
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"[✓] Ping/Test Response: {resp}")

            # 2. Get instantaneous order book snapshot for BTC-PERPETUAL
            ob_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "public/get_order_book",
                "params": {
                    "instrument_name": "BTC-PERPETUAL",
                    "depth": 5
                }
            }
            await ws.send(json.dumps(ob_req))
            ob_resp = await asyncio.wait_for(ws.recv(), timeout=5)
            ob_data = json.loads(ob_resp)
            
            if "result" in ob_data:
                res = ob_data["result"]
                best_bid = res.get("best_bid_price")
                best_ask = res.get("best_ask_price")
                spread = best_ask - best_bid if (best_ask and best_bid) else None
                bids = res.get("bids", [])[:3]
                asks = res.get("asks", [])[:3]
                print("\n" + "=" * 60)
                print("DERIBIT BTC-PERPETUAL LIVE ORDER BOOK SNAPSHOT:")
                print("=" * 60)
                print(f"Instrument:     {res.get('instrument_name')}")
                print(f"Best Bid Price: ${best_bid:,.2f}")
                print(f"Best Ask Price: ${best_ask:,.2f}")
                print(f"Bid-Ask Spread: ${spread:.2f}")
                print(f"Index Price:    ${res.get('index_price', 0):,.2f}")
                print(f"Top 3 Bids:     {bids}")
                print(f"Top 3 Asks:     {asks}")
                print("=" * 60)

            # 3. Subscribe to public unauthenticated streams
            sub_msg = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "public/subscribe",
                "params": {
                    "channels": [
                        "book.BTC-PERPETUAL.none.10.100ms",
                        "ticker.BTC-PERPETUAL.100ms",
                        "trades.BTC-PERPETUAL.100ms"
                    ]
                }
            }
            print("\n[*] Subscribing to public channels: book.BTC-PERPETUAL.none.10.100ms, ticker.BTC-PERPETUAL.100ms, trades.BTC-PERPETUAL.100ms ...")
            await ws.send(json.dumps(sub_msg))
            
            sub_ack = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"[✓] Subscription Acknowledged: {sub_ack}")

            print("\n[*] Listening for incoming live ticks (capturing 5 live events)...")
            event_count = 0
            while event_count < 5:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(msg_raw)
                if "params" in msg and "data" in msg["params"]:
                    channel = msg["params"].get("channel", "")
                    data = msg["params"]["data"]
                    event_count += 1
                    print(f"\n[Live Event #{event_count} from {channel}]")
                    if "book" in channel:
                        bids = data.get("bids", [])[:2]
                        asks = data.get("asks", [])[:2]
                        print(f"  Timestamp: {data.get('timestamp')} | Bids: {bids} | Asks: {asks}")
                    elif "ticker" in channel:
                        print(f"  Price: ${data.get('last_price')} | Best Bid: ${data.get('best_bid_price')} | Best Ask: ${data.get('best_ask_price')} | Mark: ${data.get('mark_price')}")

            print("\n[✓] Live data feed successfully received and verified from Deribit WebSocket!")
            return True

    except Exception as e:
        print(f"\n[✗] Deribit Feed Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_deribit_feed())
    sys.exit(0 if success else 1)
