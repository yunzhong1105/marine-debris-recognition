# 2026 海廢影像辨識國際 AI 競賽

物件偵測，34 類海洋廢棄物。評分為 **COCO mAP@0.5**（34 類算術平均），初賽需達 **0.6** 才能晉級。

---

## 1. 環境

訓練跑在 **Windows 原生**，不是 WSL（WSL 只分配到 7GB RAM，Windows 有 16GB）。

```powershell
cd C:\Python_workspace\marine-debris-recognition
.\.venv\Scripts\Activate.ps1
```

提示字元變成 `(.venv) PS C:\...>` 就代表進去了，之後 `python` 直接指向 venv 內的直譯器。

若 PowerShell 擋執行原則：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

用 cmd 的話改成 `.\.venv\Scripts\activate.bat`。離開環境打 `deactivate`。

| 元件 | 版本 | 備註 |
|---|---|---|
| Python | 3.12.10 | WSL 內的 3.14 太新，PyTorch 沒有對應 wheel |
| PyTorch | 2.9.1+cu126 | RTX 4060 是 sm_89；不用更新的 cu128/129，穩定優先 |
| Ultralytics | 8.4.115 | **editable install**，掛在 `vendor/ultralytics/`，pin 在 tag |

`vendor/ultralytics` 是 editable install，代表 `import ultralytics` 讀到的是那份原始碼，改了立刻生效、不需重裝。版本 pin 在 tag `v8.4.115`，因為 ultralytics 幾乎每天發版（三天內 8.4.110→115），比賽期間讓它浮動會導致分數無法重現。

### 在新機器上完整安裝

`vendor/ultralytics` 是 **git submodule**，記錄了確切的 commit（比 tag 更嚴格——tag 是可以被移動的）。所以 clone 時要加 `--recurse-submodules`：

```powershell
git clone --recurse-submodules https://github.com/yunzhong1105/marine-debris-recognition.git
cd marine-debris-recognition

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e vendor\ultralytics
```

已經 clone 但忘了加 `--recurse-submodules`（此時 `vendor/ultralytics` 會是空目錄）：

```powershell
git submodule update --init --recursive
```

資料集不在版控裡，要另外取得 `data/train_dataset/`，然後：

```powershell
python src\coco_to_yolo.py
python src\make_split.py --folds 5
```

### 搬動或改名專案目錄之後

editable install 是用絕對路徑註冊的，目錄一改名 `import ultralytics` 就會失敗。要重跑：

```powershell
pip install -e vendor\ultralytics      # 重新註冊新路徑
python src\make_split.py --folds 5     # 重生成 splits 與 configs 內的絕對路徑
```

`src/make_split.py` 會從自己的檔案位置推導專案根目錄，所以不需要手改任何設定。

---

## 2. 目錄結構

```
marine-debris-recognition/
├── .venv/                       Python 3.12 虛擬環境
├── data/
│   ├── train_dataset/
│   │   ├── images/              15,127 張原始影像（不要動）
│   │   ├── labels/              YOLO txt，由 coco_to_yolo.py 產生
│   │   └── train_label.json     官方 COCO 標註
│   └── splits/                  fold{0..4}_{train,val}.txt，影像路徑清單
├── configs/
│   ├── classes.yaml             34 類名稱
│   └── fold{0..4}.yaml          Ultralytics dataset config
├── src/
│   ├── coco_to_yolo.py          COCO JSON → YOLO txt
│   ├── make_split.py            5-fold 分層切分 + 產生 dataset config
│   └── train.py                 訓練入口
├── vendor/ultralytics/          pin 住的 ultralytics 原始碼
├── patches/                     對 vendor 的改動（目前為空）
└── runs/                        訓練輸出
```

**為什麼 `labels/` 一定要跟 `images/` 同層**：Ultralytics 找標註的方式是把影像路徑裡的 `/images/` 字串換成 `/labels/`，再把副檔名換成 `.txt`。這是硬編碼的慣例，資料夾名稱不能改。

**為什麼用檔案清單而不是分資料夾**：一般 YOLO 教學會叫你把影像複製成 `images/train/` 和 `images/val/`。這裡不這麼做——15k 張 JPEG 複製一份很浪費，而且要跑 5-fold 就得複製 5 次。改成 `fold0_train.txt` 每行一個影像絕對路徑，切分只是換一個文字檔。

---

## 3. 資料前處理（已完成，重跑才需要）

```powershell
python src\coco_to_yolo.py        # 約 4 分鐘，寫 15,127 個小檔到 /mnt/c 較慢
python src\make_split.py --folds 5
```

**`coco_to_yolo.py`** 把 COCO 的 `[x, y, w, h]`（絕對像素、左上角）轉成 YOLO 的 `cls cx cy w h`（正規化、中心點）。官方 `category_id` 本來就是連續的 0–33，直接對應 YOLO class id，不需要重新編號。轉檔結果：32,189 個框全數保留，零裁切、零退化框、零缺圖。

**`make_split.py`** 做 5-fold **iterative stratification**（Sechidis et al. 2011）。不能用隨機切分的原因：`aluminum_packaging` 全資料集只有 13 個框，隨機切會讓某些 fold 分到 0 個，該類的 AP 變成擲硬幣，而它佔 mAP 的 1/34 = 2.94%，等於憑空產生 ±3 分的雜訊，蓋過真正的實驗訊號。

分層後的結果是 13 個框被切成 **3/3/3/2/2**，已是最佳分佈；其餘 33 類每折都 ≥11 個。這個演算法是決定性的，重跑結果完全一致。

---

## 4. 訓練

```powershell
python src\train.py                                    # baseline：yolo26m @1024，fold 0，100 epochs
python src\train.py --fold 1                           # 換一折
python src\train.py --model yolo26n.pt --imgsz 640 --name nano   # 輕量化獎用
python src\train.py --epochs 1 --fraction 0.05 --name smoke      # 快速確認能跑
python src\train.py --name yolo26m_1024_fold0 --resume           # 中斷續跑
```

### 參數為什麼是這些

| 參數 | 值 | 理由 |
|---|---|---|
| `--batch` | **4** | 訓練峰值 6.09GB / 可用 7.35GB。**6 會 OOM。** |
| `--workers` | **4** | 8 會撐爆 16GB 主記憶體——mosaic 增強每個 sample 要組 2048×2048 緩衝 |
| `--imgsz` | 1024 | 資料裡有菸蒂、吸管、碎片，640 抓不到 |
| `amp` | True | 8GB 要塞 1024px 的必要條件 |
| `cache` | False | 15k 張 1024px 影像放不進 16GB RAM |

`train.py` 內含一個 **`ValBatchCappedTrainer`** 子類，覆寫驗證用的 batch size。Ultralytics 上游在 `trainer.py:293` 把驗證 dataloader 設成 `batch_size * 2`，在這台機器上會 OOM：訓練本身已用掉 6.1GB，驗證再要求 batch 8 就爆了。

這個錯誤特別陰險——**它跟記憶體碎片累積有關，短的 smoke test 會通過，跑了一兩個小時才在驗證階段掛掉**。實測 30 個 iteration 的 smoke test 沒事，1512 個 iteration 的半 epoch 就死。（Windows 不支援 `expandable_segments`，那條常見解法在這裡無效。）

改成子類而非直接改 `vendor/`，是為了之後升級 ultralytics 時不用處理衝突。

### 效能實測

| 項目 | 數值 |
|---|---|
| 訓練速度 | **2.7 it/s**（batch 4 @ 1024） |
| 一個 epoch | 3,024 iterations ≈ **19 分鐘** + 驗證約 2 分鐘 |
| 100 epochs | **約 35 小時** |

這是重要的規劃數字：一次完整訓練要一天半。初賽只有三週，代表最多跑 10 次左右完整實驗，**參數不能亂試**。建議前期用 `--fraction 0.3` 加短 epoch 快速篩方向，選定後才跑完整訓練。

### 輸出

```
runs/<name>/
├── weights/best.pt      驗證分數最好的權重
├── weights/last.pt      最後一個 epoch，--resume 用這個
├── results.csv          每個 epoch 的 loss 與指標
├── results.png          訓練曲線
├── labels.jpg           類別分佈與框的分佈
├── train_batch*.jpg     增強後的訓練影像，用來確認 augmentation 沒壞掉
└── val_batch*_pred.jpg  驗證預測 vs 標註對照
```

---

## 5. 尚未完成

- **`src/predict.py`** — 產生 submission.csv，欄位 `image_filename, label_id, x, y, w, h, confidence`
- **`src/evaluate.py`** — 用 pycocotools 算 mAP@0.5，對齊官方計分
- **`end2end` A/B** — YOLO26 預設走 one-to-one head（無 NMS），對 mAP 可能不利，要實測比較

### 產生 submission 時的關鍵陷阱

推論務必用 **`conf=0.001`、`max_det=300`**，不要用 Ultralytics 預設的 `conf=0.25`。

AP 是 PR 曲線下的面積，計算方式是把預測依信心分數排序後逐一納入。砍掉低分預測等於把曲線在中途截斷，面積直接少一塊。對 mAP 而言，多送一個低分錯誤框的代價幾乎是零，但漏掉一個物件會永久損失 recall——**這是不對稱的，所以要盡量多送**。

`conf=0.25` 是給人看圖用的，不是給計分用的。用預設值產生 submission 會無謂損失大量分數。
