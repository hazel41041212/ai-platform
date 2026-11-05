# 校園 FAQ 智慧查詢 Campus FAQ Search

用 Python 呼叫 Google Gemini 的 **Embedding（文字向量）** 功能做語意搜尋：使用者用自己的話提問，程式從校園 FAQ 題庫中找出意思最接近的 3 題，並和傳統關鍵字比對做比較。結果由 GitHub Actions 自動部署到 GitHub Pages 作品頁。

- 作品頁：https://<你的帳號>.github.io/campus-faq-search/
- 使用的 LLM 能力：Embedding（文字向量 + 餘弦相似度）

## 功能

| 功能 | 說明 |
|---|---|
| 語意搜尋 | 問題與題庫都轉成向量，用餘弦相似度取前 3 名 |
| 自動評估 | 每個測試問題事先標好正確答案，計算 Top-1 正確率、Top-3 命中率 |
| 對照組 | 不用 AI 的關鍵字比對（兩字一組的 Jaccard 相似度），比較兩者差異 |
| 題庫外偵測觀察 | 放入題庫沒有答案的問題，觀察相似度是否明顯較低 |
| 現場查詢 | `uv run main.py --ask "你的問題"` |

## 專案結構

```
.
├── .github/workflows/deploy.yml  # 自動部署 GitHub Pages
├── data/
│   ├── faq.json                  # 輸入：FAQ 題庫
│   └── queries.json              # 輸入：測試問題與正確答案
├── site/
│   ├── index.html                # 作品頁
│   └── results.json              # 輸出：程式產生的查詢結果
├── main.py                       # 主程式
├── pyproject.toml                # uv 套件設定
├── .env.example                  # 金鑰範例（不含真實金鑰）
└── .gitignore                    # 排除 .env
```

## 執行步驟

1. 到 [Google AI Studio](https://aistudio.google.com/) 取得免費 API key
2. 安裝套件：
   ```bash
   pip install uv   # 若尚未安裝 uv
   uv sync
   ```
3. 建立 `.env` 並填入金鑰：
   ```bash
   cp .env.example .env
   ```
4. 執行：
   ```bash
   uv run main.py                         # 完整評估，產生 site/results.json
   uv run main.py --ask "宿舍可以煮泡麵嗎"  # 現場問一題
   uv run main.py --list-models           # 出現 404 時，查詢可用的模型名稱
   ```
5. 將 `site/results.json` commit 並 push，GitHub Actions 會自動部署作品頁。

## 更換題庫與測試問題

`data/faq.json` 每題格式：

```json
{ "id": "F01", "category": "選課", "question": "問題", "answer": "答案" }
```

`data/queries.json` 每題格式（`expected_id` 填正確 FAQ 的編號；題庫裡沒有答案的問題填 `null`）：

```json
{ "query": "用自己的話問的問題", "expected_id": "F01" }
```

## 安全性

API 金鑰只放在 `.env`，該檔已列入 `.gitignore`，不會進入 repo 或 commit 歷史。GitHub Actions 只部署網頁，不呼叫 AI，因此不需要在 GitHub 設定金鑰。

## AI 使用說明

（請依實際情況填寫：用了哪些 AI 工具、用在哪裡、自己修改或驗證了什麼）
