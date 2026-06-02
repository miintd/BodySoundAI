#!/bin/bash

set -e
SCRIPT="python parse_auc_from_log.py"

# echo ""
# echo "============================================================"
# echo " 1) logs_bts/COVID  — AudioBert models"
# echo "============================================================"
# $SCRIPT --folder logs_bts/COVID --pattern AudioBert

echo ""
echo "============================================================"
echo " 2) logs_llm_fix_seed/COVID  — GPT2 models"
echo "============================================================"
$SCRIPT --folder logs_llm_fix_seed/COVID --pattern out

# echo ""
# echo "============================================================"
# echo " 3) logs_clap  — CLAP concat models"
# echo "============================================================"
# $SCRIPT --folder logs_clap --pattern clap_concat

# echo ""
# echo "============================================================"
# echo " BONUS: một CSV tổng hợp tất cả folder"
# echo "============================================================"
# $SCRIPT \
#   --folders logs_bts/COVID logs_llm_fix_seed/COVID logs_clap \
#   --output_dir results_combined

# echo ""
# echo "Done! Các file CSV được lưu trong từng folder tương ứng,"
# echo "và combined CSV ở results_combined/merged_auc_all_folders.csv"