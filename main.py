"""校園 FAQ 智慧查詢：用 Gemini 的 embedding（文字向量）做「語意搜尋」。

和關鍵字搜尋不同，語意搜尋看的是「意思像不像」，
例如「書快到期了可以延長嗎」雖然沒有出現「續借」兩個字，也能找到「借書可以借多久？」這題。

用法：
    uv run main.py                          # 跑完整評估，產生 site/results.json（作品頁會讀這個檔）
    uv run main.py --ask "宿舍可以煮泡麵嗎"   # 現場問一題，印出最相似的 3 題 FAQ
    uv run main.py --list-models            # 列出你的帳號可用的 embedding 模型（遇到 404 時用）

輸入：data/faq.json（題庫）、data/queries.json（測試問題）
輸出：site/results.json
"""

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------- 設定 ----------
load_dotenv()  # 從 .env 讀取 GEMINI_API_KEY（.env 不會進 repo）
MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
FAQ_FILE = Path("data/faq.json")
QUERY_FILE = Path("data/queries.json")
OUTPUT_FILE = Path("site/results.json")
TOP_K = 3          # 每個問題列出最相似的前 3 題
BATCH_SIZE = 50    # 一次送多筆文字去轉向量，減少 API 呼叫次數


# ---------- 1. 呼叫 Gemini：把文字轉成向量 ----------
def embed(client: genai.Client, texts: list[str], kind: str) -> list[list[float]]:
    """把一批文字轉成向量。kind = "query"（使用者的問題）或 "document"（題庫內容）。

    同一個模型要用「同一套規則」處理問題與題庫，比較出來的分數才有意義：
    - gemini-embedding-001：用 task_type 告訴模型這段文字是「查詢」還是「文件」
    - gemini-embedding-2  ：官方建議在文字前面加前綴（prefix）
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        if "embedding-2" in MODEL:
            prefix = "task: search result | query: " if kind == "query" else "title: none | text: "
            chunk = [prefix + t for t in chunk]
            config = None
        else:
            task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
            config = types.EmbedContentConfig(task_type=task)

        for attempt in range(3):  # 遇到免費額度的速率限制（429）就等一下再試
            try:
                result = client.models.embed_content(model=MODEL, contents=chunk, config=config)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    print("   超過每分鐘上限，等 30 秒後重試…")
                    time.sleep(30)
                else:
                    raise
        vectors.extend(e.values for e in result.embeddings)
    return vectors


# ---------- 2. 計算相似度 ----------
def cosine(a: list[float], b: list[float]) -> float:
    """餘弦相似度：兩個向量方向越接近，值越接近 1，代表意思越像。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def search(q_vec: list[float], faq: list[dict], faq_vecs: list[list[float]], k: int = TOP_K) -> list[dict]:
    scored = [
        {"id": f["id"], "category": f["category"], "question": f["question"],
         "answer": f["answer"], "score": round(cosine(q_vec, v), 4)}
        for f, v in zip(faq, faq_vecs)
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]


# ---------- 3. 對照組：傳統關鍵字比對（不使用 AI） ----------
def bigrams(text: str) -> set[str]:
    """把句子切成「每兩個字一組」，例如「借書期限」→ {借書, 書期, 期限}。"""
    s = re.sub(r"[^\w]", "", text)  # 去掉標點與空白
    return {s[i:i + 2] for i in range(len(s) - 1)}


def keyword_top1(query: str, faq: list[dict]) -> dict:
    qb = bigrams(query)
    best, best_score = faq[0], -1.0
    for f in faq:
        fb = bigrams(f["question"] + f["answer"])
        score = len(qb & fb) / len(qb | fb) if qb | fb else 0.0  # Jaccard 相似度
        if score > best_score:
            best, best_score = f, score
    return {"id": best["id"], "question": best["question"], "score": round(best_score, 4)}


# ---------- 主流程 ----------
def doc_text(f: dict) -> str:
    return f"{f['question']}\n{f['answer']}"


def load_faq_with_vectors(client: genai.Client) -> tuple[list[dict], list[list[float]]]:
    faq = json.loads(FAQ_FILE.read_text(encoding="utf-8"))
    print(f"題庫共 {len(faq)} 題，轉成向量中（模型：{MODEL}）…")
    return faq, embed(client, [doc_text(f) for f in faq], "document")


def run_evaluation(client: genai.Client) -> None:
    faq, faq_vecs = load_faq_with_vectors(client)
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    print(f"測試問題共 {len(queries)} 題，轉成向量中…")
    q_vecs = embed(client, [q["query"] for q in queries], "query")
    print(f"向量維度：{len(faq_vecs[0])}\n")

    items = []
    for q, v in zip(queries, q_vecs):
        matches = search(v, faq, faq_vecs)
        expected = q.get("expected_id")  # null 代表「題庫裡沒有答案」的問題
        ids = [m["id"] for m in matches]
        hit_rank = ids.index(expected) + 1 if expected in ids else None
        kw = keyword_top1(q["query"], faq)
        items.append({
            "query": q["query"],
            "expected_id": expected,
            "hit_rank": hit_rank,
            "matches": matches,
            "keyword_top1": kw,
            "keyword_correct": kw["id"] == expected if expected else None,
        })
        mark = "—" if expected is None else ("✓" if hit_rank == 1 else f"第{hit_rank}名" if hit_rank else "✗")
        print(f"{mark:>4}  {q['query']}  →  {matches[0]['id']} {matches[0]['question']}（{matches[0]['score']:.3f}）")

    labeled = [it for it in items if it["expected_id"]]
    unlabeled = [it for it in items if not it["expected_id"]]
    n = len(labeled) or 1
    metrics = {
        "labeled": len(labeled),
        "top1_accuracy": round(sum(it["hit_rank"] == 1 for it in labeled) / n, 4),
        "top3_hit_rate": round(sum(it["hit_rank"] is not None for it in labeled) / n, 4),
        "keyword_top1_accuracy": round(sum(bool(it["keyword_correct"]) for it in labeled) / n, 4),
        "avg_top1_score_in_scope": round(sum(it["matches"][0]["score"] for it in labeled) / n, 4),
        "avg_top1_score_out_of_scope": (
            round(sum(it["matches"][0]["score"] for it in unlabeled) / len(unlabeled), 4) if unlabeled else None
        ),
    }

    taipei = timezone(timedelta(hours=8))
    output = {
        "model": MODEL,
        "generated_at": datetime.now(taipei).strftime("%Y-%m-%d %H:%M"),
        "dimension": len(faq_vecs[0]),
        "faq_count": len(faq),
        "query_count": len(queries),
        "metrics": metrics,
        "items": items,
        "faq": faq,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n語意搜尋 Top-1 正確率：{metrics['top1_accuracy']:.0%}")
    print(f"語意搜尋 Top-3 命中率：{metrics['top3_hit_rate']:.0%}")
    print(f"關鍵字比對 Top-1 正確率：{metrics['keyword_top1_accuracy']:.0%}（對照組）")
    print(f"\n完成！已寫入 {OUTPUT_FILE}")


def ask_one(client: genai.Client, question: str) -> None:
    faq, faq_vecs = load_faq_with_vectors(client)
    q_vec = embed(client, [question], "query")[0]
    print(f"\n問題：{question}\n")
    for rank, m in enumerate(search(q_vec, faq, faq_vecs), start=1):
        print(f"{rank}. [{m['id']}｜{m['category']}] {m['question']}  相似度 {m['score']:.3f}")
        print(f"   {m['answer']}\n")


def list_models(client: genai.Client) -> None:
    print("可用的 embedding 模型：")
    for m in client.models.list():
        if "embedContent" in (m.supported_actions or []):
            print(" ", m.name.removeprefix("models/"))


def main() -> None:
    parser = argparse.ArgumentParser(description="校園 FAQ 智慧查詢")
    parser.add_argument("--ask", help="直接問一個問題")
    parser.add_argument("--list-models", action="store_true", help="列出可用的 embedding 模型")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("找不到 GEMINI_API_KEY，請先建立 .env（可參考 .env.example）")
    client = genai.Client()  # 自動讀取環境變數 GEMINI_API_KEY

    if args.list_models:
        list_models(client)
    elif args.ask:
        ask_one(client, args.ask)
    else:
        run_evaluation(client)


if __name__ == "__main__":
    main()
