Towards Prompt-Robust Machine-Generated Text Detection
----

## 🛠️ Installation

### Requirements
- Python 3.10.8
- PyTorch 2.7.0
- CUDA-compatible GPU (experiments conducted on H20-NVLink with 96GB memory)

### Setup
```bash
bash setup.sh
```

(Finished in Week 1 and try to run `diverse.sh` successfully on our method and several benchmark)
- Hint: 1. download data from https://github.com/ranhli/l2r_data; 2. to run our method, I recommend to first run the rewrite-based method and then the ada-rewrite-based method. 

### Tasks

Experiments on

- data generated from GPT-5 (Finished in Week 2)
  - Hint: 1. update `data_builder_prompt.py` to support gpt-5; 2. change the input of `scripts/data_builder_prompt.py --base_model_name 'gpt5'`.  
- data generated from the "generation" task (see https://arxiv.org/pdf/2412.10432，section 3.1) (Finished in Week 3)
  - Hint: 1. update `data_builder_prompt.py` to support the generation task; 2. change the input of `scripts/data_builder_prompt.py --tasks 'generate'`
- an additional baseline on diverse datasets (e.g., https://github.com/baoguangsheng/glimpse) (Finished in Week 4)
  - Hint: 1. take a look on the output structure of the current implementation; 2. update the code in https://github.com/baoguangsheng/glimpse to adopt the output structure; 3. add the newly implemented method in the `report_diverse_results` function in `report_results.py` 

### Reproduce guidance


- `diverse.sh`: generate Table 1, Tables B1-B4
- `blackbox_prompt.sh`: generate Table 2 and Table 3
- `attack_rewrite`: Figure 4

After running the above code, please use `python script/report_results.py` to see the results. Use either the `report_black_prompt_results` or the `report_diverse_results` functions.

If you have any question, please feel free to contact Jin Zhu via WeChat or email.