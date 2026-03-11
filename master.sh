#!/bin/bash

# Danh sách các file cần quét
FILES=(
    "src/benchmark/RespLLM/model.py"
    "/home/tran-dam-quoc-khanh/Documents/DIEM_MY/OPERA/src/benchmark/RespLLM/RespLLM.py"
)

echo "--- Checking tokens ---"

for FILE in "${FILES[@]}"; do
    # Kiểm tra file có tồn tại không trước khi quét
    if [ -f "$FILE" ]; then
        if grep -q "redacted" "$FILE"; then
            echo "--------------------------------------------------------"
            echo "have not include token in the folowing files:"
            echo "-> $FILE"
            echo "--------------------------------------------------------"
            echo "You have to write token like the tutorial."
            # echo "Vui lòng kiểm tra lại os.environ['WANDB_API_KEY'] hoặc file .env"
            exit 1 # Ngắt toàn bộ script bash với mã lỗi
        fi
    else
        echo "Cảnh báo: Không tìm thấy file $FILE để quét."
    fi
done

echo "Token valid. Continue."

bash scripts/clone_data.sh
source scripts/setup_env.sh
bash scripts/run_audio_label.sh
bash scripts/run_wavelet.sh