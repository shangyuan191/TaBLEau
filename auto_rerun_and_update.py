import re
import subprocess
import shutil

RESULT_FILE = "./results/dataset_size_all_task_type_all_feature_type_all_models_trompt_gnn_stages_all_epochs_300_0.05_0.15_0.8.txt"
NEW_FILE = RESULT_FILE + ".new"
BACKUP_FILE = RESULT_FILE + ".bak"

def rerun_case(dataset, gnn_stage):
    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--models", "trompt",
        "--gnn_stages", gnn_stage,
        "--epochs", "300"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"results_file_name: (.+\.txt)", result.stdout)
    if not m:
        print("無法取得新結果檔名，請檢查 main.py 輸出")
        return None
    new_file = "./results/" + m.group(1)
    with open(new_file, "r", encoding="utf-8") as f:
        new_text = f.read()
    # 只抓對應 dataset/gnn_stage 的 trompt區塊
    pat = re.compile(
        rf"dataset: {re.escape(dataset)}\n  模型: trompt\n((?:    GNN階段: .+\n(?:[^\n]*\n)+)+)",
        re.MULTILINE)
    m = pat.search(new_text)
    if not m:
        print(f"無法在新結果檔案中找到 {dataset} trompt 區塊")
        return None
    block = m.group(1)
    gnn_pat = re.compile(
        rf"(    GNN階段: {re.escape(gnn_stage)}\n(?:[^\n]*\n)+?)(?=    GNN階段:|\Z)", re.MULTILINE)
    m2 = gnn_pat.search(block)
    if not m2:
        print(f"無法在新結果中找到 GNN階段: {gnn_stage}")
        return None
    return m2.group(1)

def main():
    shutil.copyfile(RESULT_FILE, BACKUP_FILE)
    print(f"已備份原始檔案到 {BACKUP_FILE}")

    with open(RESULT_FILE, "r", encoding="utf-8") as fin, open(NEW_FILE, "w", encoding="utf-8") as fout:
        dataset = None
        model = None
        gnn_stage = None
        block_lines = []
        in_gnn_block = False

        for line in fin:
            dataset_m = re.match(r"^dataset: (.+)$", line)
            model_m = re.match(r"^  模型: (.+)$", line)
            gnn_m = re.match(r"^    GNN階段: (.+)$", line)

            # 新增這段：遇到新dataset或model時，先處理上一個GNN block
            if (dataset_m or model_m) and block_lines:
                block_str = ''.join(block_lines)
                if ("Best val metric: inf" in block_str or "Best test metric: inf" in block_str or "錯誤:" in block_str):
                    print(f"重跑 {dataset} {gnn_stage}")
                    new_block = rerun_case(dataset, gnn_stage)
                    if new_block:
                        fout.write(new_block)
                    else:
                        fout.write(block_str)
                else:
                    fout.write(block_str)
                block_lines = []

            if dataset_m:
                dataset = dataset_m.group(1)
                fout.write(line)
                continue
            if model_m:
                model = model_m.group(1)
                fout.write(line)
                continue
            if gnn_m:
                # ...原本的GNN block處理...
                if block_lines:
                    block_str = ''.join(block_lines)
                    if ("Best val metric: inf" in block_str or "Best test metric: inf" in block_str or "錯誤:" in block_str):
                        print(f"重跑 {dataset} {gnn_stage}")
                        new_block = rerun_case(dataset, gnn_stage)
                        if new_block:
                            fout.write(new_block)
                        else:
                            fout.write(block_str)
                    else:
                        fout.write(block_str)
                    block_lines = []
                gnn_stage = gnn_m.group(1)
            # 收集 GNN block 內容
            if gnn_m or (block_lines and not dataset_m and not model_m):
                block_lines.append(line)
            else:
                if not gnn_m:
                    fout.write(line)

        # 檔案結尾最後一個 GNN block
        if block_lines:
            block_str = ''.join(block_lines)
            if ("Best val metric: inf" in block_str or "Best test metric: inf" in block_str or "錯誤:" in block_str):
                print(f"重跑 {dataset} {gnn_stage}")
                new_block = rerun_case(dataset, gnn_stage)
                if new_block:
                    fout.write(new_block)
                else:
                    fout.write(block_str)
            else:
                fout.write(block_str)

    print(f"新檔案已寫入 {NEW_FILE}")

if __name__ == "__main__":
    main()





# dataset: kaggle_Real_Estate_DataSet
#   模型: excelformer
#     GNN階段: none
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: start
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: materialize
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: encoding
#           Best val metric: 0.910468339920044
#           Best test metric: 0.8666666746139526
#           早停輪數: 21
#           GNN早停輪數: 0
#           耗時: 2.42 秒
#     GNN階段: columnwise
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: decoding
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
# dataset: openml_Indian_Liver_Patient_Patient_Records_KFolds_5folds
#   模型: excelformer
#     GNN階段: none
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: start
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: materialize
#           Best val metric: 0.7358729839324951
#           Best test metric: 0.9153439402580261
#           早停輪數: 17
#           GNN早停輪數: 0
#           耗時: 2.17 秒
#     GNN階段: encoding
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: columnwise
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: decoding
#           Best val metric: 0.742222249507904
#           Best test metric: 0.7354497909545898
#           早停輪數: 15
#           GNN早停輪數: 98
#           耗時: 2.93 秒
# dataset: openml_Diabetes_Data_Set
#   模型: excelformer
#     GNN階段: none
#           Best val metric: 0.7194079160690308
#           Best test metric: 0.6485714316368103
#           早停輪數: 20
#           GNN早停輪數: 0
#           耗時: 2.35 秒
#     GNN階段: start
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: materialize
#           Best val metric: 0.7072368264198303
#           Best test metric: 0.8400000333786011
#           早停輪數: 25
#           GNN早停輪數: 0
#           耗時: 3.66 秒
#     GNN階段: encoding
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: columnwise
#           Best val metric: 0.8154605627059937
#           Best test metric: 0.8028571605682373
#           早停輪數: 26
#           GNN早停輪數: 0
#           耗時: 3.40 秒
#     GNN階段: decoding
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
# dataset: kaggle_Penguins_Classified
#   模型: excelformer
#     GNN階段: none
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: start
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: materialize
#           Best val metric: 0.9970015287399292
#           Best test metric: 1.0
#           早停輪數: 18
#           GNN早停輪數: 0
#           耗時: 2.38 秒
#     GNN階段: encoding
#           Best val metric: 1.0
#           Best test metric: 1.0
#           早停輪數: 17
#           GNN早停輪數: 0
#           耗時: 2.00 秒
#     GNN階段: columnwise
#           Best val metric: inf
#           Best test metric: inf
#           錯誤: ERROR
#           耗時: 3.95 秒
#     GNN階段: decoding
#           Best val metric: 0.995502233505249
#           Best test metric: 0.9375
#           早停輪數: 19
#           GNN早停輪數: 11
#           耗時: 5.86 秒