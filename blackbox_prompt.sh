#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
python_path=/Users/j.zhu.7@bham.ac.uk/miniconda3/envs/dgpt/bin/python
exp_path=exp_prompt
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# gpu_device='cuda:0'
# gpu_device='cuda:1'
gpu_device='cuda'

source_models="gpt-4o gemini-2.5-flash claude-3-5-haiku"
datasets="xsum squad writing"
tasks="expand rewrite polish"

# preparing dataset
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      echo date, Preparing dataset ${D}_${M}_${task} ...
      $python_path scripts/data_builder_prompt.py \
        --dataset $D \
        --task $task \
        --n_samples 100 \
        --base_model_name $M \
        --output_file $data_path/${D}_${M}_${task} \
        --do_temperature  --temperature 0.8
    done
  done
done

settings='gemma-9b:gemma-9b-instruct'
scoring_models="gemma-9b-instruct"

# evaluate the rewrite-based method
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      for M1 in $scoring_models; do
        echo `date`, Evaluating Methods on ${D}_${M} ...
        python scripts/detect_rewrite2.py --base_model_name $M1 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
        python scripts/detect_revised.py --base_model_name $M1 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
        python scripts/detect_revised2.py --base_model_name $M1 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
      done
    done
  done
done

# evaluate FastDetectGPT and Binoculars
for M in $source_models; do
  for task in $tasks; do
    for D in $datasets; do
      for S in $settings; do
        IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
        echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
        python scripts/fast_detect_gpt.py --sampling_model_name $M1 --scoring_model_name $M2 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --discrepancy_analytic --device $gpu_device
        python scripts/detect_bino.py --model1_name $M1 --model2_name $M2 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
      done
    done
  done
done

# evaluate fast baselines
for M in $source_models; do
  for task in $tasks; do
    for D in $datasets; do
      for M2 in $scoring_models; do
        echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
        python scripts/baselines.py --scoring_model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
        python scripts/detect_lrr.py --scoring_model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
        python scripts/detect_phd.py --model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --solver 'MLE' --device $gpu_device
      done
    done
  done
done

# evaluate RADIAR 
for M in $source_models; do
  for D in $datasets; do
    train_dataset=""
    for D1 in $datasets; do
      if [ "$D1" = "$D" ]; then
        continue  # 排除与测试集相同的 dataset
      fi

      for T1 in $tasks; do
        if [ -z "$train_dataset" ]; then
          train_dataset="${data_path}/${D1}_${M}_${T1}"
        else
          train_dataset="${train_dataset}&${data_path}/${D1}_${M}_${T1}"
        fi
      done
    done

    for task in $tasks; do
      python scripts/detect_raidar.py --train_dataset ${train_dataset} --eval_dataset $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task}
    done
  done
done

# evaluate the ada-rewrite-based and ImBD
trained_model_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
for M2 in $scoring_models; do
  for M in $source_models; do
    for D in $datasets; do
      train_dataset=""
      for D1 in $datasets; do
        if [ "$D1" = "$D" ]; then
          continue  # 排除与测试集相同的 dataset
        fi

        for T1 in $tasks; do
          if [ -z "$train_dataset" ]; then
          train_dataset="${data_path}/${D1}_${M}_${T1}"
          else
          train_dataset="${train_dataset}&${data_path}/${D1}_${M}_${T1}"
          fi
        done
      done
      echo "Train data: $train_dataset"
      python scripts/detect_rewrite_ada.py --datanum 500 --base_model $M2 --train_dataset ${train_dataset} --save_trained
      # python scripts/detect_ImBD_task.py --datanum 500 --base_model $M2 --train_dataset ${train_dataset} --save_trained

      for task in $tasks; do
          python scripts/detect_rewrite_ada.py --eval_only --base_model $M2 --eval_dataset $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --from_pretrained $trained_model_path
          # python scripts/detect_ImBD_task.py --eval_only --base_model $M2 --eval_dataset $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --from_pretrained $trained_model_path
      done
    done
  done
done

# evaluate RADAR
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      echo `date`, Evaluating RADAR on ${D}_${M}_${task}  ...
      python scripts/detect_radar.py --dataset $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task}
    done
  done
done

# evaluate RoBerta
supervised_models="roberta-large-openai-detector"
for task in $tasks; do
  for D in $datasets; do
    for M in $source_models; do
      for SM in $supervised_models; do
        echo `date`, Evaluating ${SM} on ${D}_${M}_${task} ...
        python scripts/supervised.py --model_name $SM --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
      done
    done
  done
done

# # evaluate DNA-GPT
# for M in $source_models; do
#     for task in $tasks; do
#         for D in $datasets; do
#             for M1 in $scoring_models; do
#                 echo `date`, Evaluating Methods on ${D}_${M} ...
#                 python scripts/dna_gpt.py --base_model_name $M1 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task} --device $gpu_device
#             done
#         done
#     done
# done

# # evaluate DetectGPT and its improvement DetectLLM
# for task in $tasks; do
#   for D in $datasets; do
#     for M in $source_models; do
#       echo `date`, Evaluating DetectGPT on ${D}_${M}_${task} ...
#       # python scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D --dataset_file $data_path/${D}_${M}_${task} --output_file $res_path/${D}_${M}_${task}
#       python scripts/detect_llm.py --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M}
#     done
#   done
# done