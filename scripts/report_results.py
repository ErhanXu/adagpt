# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import argparse
import json
import numpy as np
from scipy.stats import norm
from itertools import chain

def save_lines(lines, file):
    with open(file, 'w') as fout:
        fout.write('\n'.join(lines))

def get_auroc(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['roc_auc']

def get_sampled_mean(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return np.mean(res['predictions']['samples'])

def get_fpr_tpr(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['fpr'], res['metrics']['tpr']

def report_main_results(args):
    datasets = {
        'xsum': 'XSum',
        'squad': 'SQuAD',
        'writing': 'WritingPrompts', 
        'yelp': 'Yelp', 
        'essay': 'Essay',
    }
    source_models = {
        # 'gpt2-xl': 'GPT-2',
        # 'opt-2.7b': 'OPT-2.7',
        # 'gpt-neo-2.7B': 'Neo-2.7',
        # 'gpt-j-6B': 'GPT-J',
        # 'gpt-neox-20b': 'NeoX', 
        'qwen-7b': 'Qwen2.5', 
        'mistralai-7b': 'Mistral', 
        'llama3-8b': 'LLaMA3',
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        'lrr': 'LRR',
        'npr': 'NPR', 
        # 'dna_gpt': 'DNAGPT',
    }
    methods2 = {
        'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'binoculars': 'Binoculars',
        # 'sampling_discrepancy_analytic': 'FastDetectGPT', 
        'fluoroscopy': 'TextFluoroscopy',
        'radar': 'RADAR',
        'imbd': 'ImBD',
        'superadadetectgpt': 'SuperAdaDetectGPT',
        'biscope': 'BiScope',
        'classification.bspline': 'AdaDetectGPT',
        # 'variance.identity': 'VarDetectGPT',
        # 'appro_variance.identity': 'ApproxVarDetectGPT',
        }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        # print('Diff', ' '.join(cols))
        relatives = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        relatives = 100 * relatives / (1.0 - np.array(results['FastDetectGPT']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative (over FastDetect)', ' '.join(relatives))

        relatives = np.array(results['AdaDetectGPT']) - np.array(results['Binoculars'])
        relatives = 100 * relatives / (1.0 - np.array(results['Binoculars']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative (over Binoculars)', ' '.join(relatives))
        ## black-box comparison
        print('>>>>>>>>>>>> Black-box comparison >>>>>>>>>>>>')
        methods3 = {
            # 'perturbation_100': 'DetectGPT',
            # 'sampling_discrepancy': 'Fast-DetectGPT'
            'sampling_discrepancy': 'FastDetectGPT', 
            # 'sampling_discrepancy_analytic': 'FastDetectGPT', 
            'classification': 'AdaDetectGPT',
            }
        filters = {
            # 'perturbation_100': '.t5-3b_gpt-neo-2.7B',
            # 'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-neo-2.7B', 
            'sampling_discrepancy': '.gpt-j-6B_gpt-neo-2.7B', 
            'classification': '.gpt-j-6B_gpt-neo-2.7B',
            # 'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-j-6B', 
            # 'sampling_discrepancy': '.gpt-j-6B_gpt-j-6B', 
            # 'classification': '.gpt-j-6B_gpt-j-6B',
        }
        results = {}
        for method in methods3:
            method_name = methods3[method]
            cols = _get_method_aurocs(dataset, method, filters[method])
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        # cols = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        # cols = [f'{col:.4f}' for col in cols]
        # print('Diff', ' '.join(cols))
        relatives = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        relatives = 100 * relatives / (1.0 - np.array(results['FastDetectGPT']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative', ' '.join(relatives))

def report_prompt_results(args):
    datasets = {
        'xsum': 'XSum',
        'squad': 'SQuAD',
        'writing': 'Writing', 
        # 'bbc': "BBC"
    }
    source_models = {
        # 'qwen-7b-instruct': 'Qwen2.5', 
        'mistralai-7b-instruct': 'Mistral', 
        'llama3-8b-instruct': 'LLaMA3',
        'gemma-9b-instruct': 'Gemma',
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        'lrr': 'LRR',
        'npr': 'NPR', 
        'dna_gpt': 'DNAGPT',
        'binoculars': 'Binoculars',
        'ide_mle': 'IntrinsicDim(MLE)',
        'ide_twonn': 'IntrinsicDim(NN)',
    }
    methods2 = {
        'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'revised_gpt2': 'RevisedGPT',
        'rewrite_gpt': 'RewriteGPT(Likelihood)',
        'revised_gpt': 'RewriteGPT(Dist)',
        # 'adarewritegpt': 'RewriteGPT(Ada1)',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        # 'adarewritegpt_mean_gap': 'RewriteGPT(Ada2)',
        'adarewritegpt_auc': 'RewriteGPT',
        # 'fluoroscopy': 'TextFluoroscopy',
        # 'biscope': 'BiScope',
        # 'classification.bspline': 'AdaDetectGPT',
        # 'superadadetectgpt': 'SuperAdaDetectGPT',
    }

    def _get_method_aurocs(dataset, method):
        cols = []
        for model in source_models:
            task_mean = []
            for case in task_list:
                result_file = f'{args.result_path}/{dataset}_{model}_{case}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
                task_mean.append(auroc)
            cols.append(np.mean(task_mean))
        return cols

    def _get_method_mean(dataset, method):
        cols = []
        for model in source_models:
            for case in task_list:
                result_file = f'{args.result_path}/{dataset}_{model}_{case}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_sampled_mean(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
        return cols

    # task_list = ['rewrite', 'polish', 'expand', 'generation']
    # task_list = ['rewrite', 'polish', 'expand', 'summary', 'generation']
    task_list = ['rewrite', 'polish', 'expand']
    headers1 = ['      '] + list(chain.from_iterable([[source_models[model]]*(len(task_list)+1) for model in source_models]))
    headers2 = ['Method'] + list(chain.from_iterable([[task for task in task_list] + ['Avg.']]*len(source_models)))
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers1))
        print(' '.join(headers2))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        # results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            # results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

    # for dataset in datasets:
    #     print('----')
    #     print(datasets[dataset])
    #     print('----')
    #     print(' '.join(headers1))
    #     print(' '.join(headers2))
    #     # basic methods
    #     for method in methods1:
    #         method_name = methods1[method]
    #         cols = _get_method_mean(dataset, method)
    #         cols = [f'{col:.4f}' for col in cols]
    #         print(method_name, ' '.join(cols))
    #     # white-box comparison
    #     # results = {}
    #     for method in methods2:
    #         method_name = methods2[method]
    #         cols = _get_method_mean(dataset, method)
    #         # results[method_name] = cols
    #         cols = [f'{col:.4f}' for col in cols]
    #         print(method_name, ' '.join(cols))


def report_black_prompt_results(args):
    datasets = {
        'xsum': 'XSum',
        'squad': 'SQuAD',
        'writing': 'Writing', 
        # 'bbc': "BBC"
    }
    source_models = {
        'claude-3-5-haiku': 'Claude', 
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini',
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        # 'lrr': 'LRR',
        # 'npr': 'NPR', 
        'dna_gpt': 'DNAGPT',
        'binoculars': 'Binoculars',
        'ide_mle': 'IntrinsicDim(MLE)',
        'ide_twonn': 'IntrinsicDim(NN)',
    }
    methods2 = {
        # 'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'revised_gpt2': 'RevisedGPT',
        'rewrite_gpt': 'RewriteGPT(Likelihood)',
        'revised_gpt': 'RewriteGPT(Dist)',
        # 'adarewritegpt': 'RewriteGPT(Ada1)',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        # 'adarewritegpt_mean_gap': 'RewriteGPT(Ada2)',
        'adarewritegpt_auc': 'RewriteGPT',
        # 'fluoroscopy': 'TextFluoroscopy',
        # 'biscope': 'BiScope',
        # 'classification.bspline': 'AdaDetectGPT',
        # 'superadadetectgpt': 'SuperAdaDetectGPT',
    }

    def _get_method_aurocs(dataset, method):
        cols = []
        for model in source_models:
            task_mean = []
            for case in task_list:
                result_file = f'{args.result_path}/{dataset}_{model}_{case}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
                task_mean.append(auroc)
            cols.append(np.mean(task_mean))
        return cols

    def _get_method_mean(dataset, method):
        cols = []
        for model in source_models:
            for case in task_list:
                result_file = f'{args.result_path}/{dataset}_{model}_{case}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_sampled_mean(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
        return cols

    # task_list = ['rewrite', 'polish', 'expand', 'generation']
    # task_list = ['rewrite', 'polish', 'expand', 'summary', 'generation']
    task_list = ['rewrite', 'polish', 'expand']
    headers1 = ['      '] + list(chain.from_iterable([[source_models[model]]*(len(task_list)+1) for model in source_models]))
    headers2 = ['Method'] + list(chain.from_iterable([[task for task in task_list] + ['Avg.']]*len(source_models)))
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers1))
        print(' '.join(headers2))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        # results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            # results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def report_diverse_results(args):
    datasets = {
        'AcademicResearch':'AcademicResearch', 
        'EducationMaterial':'EducationMaterial',
        'FoodCusine':'FoodCusine',
        'MedicalText':'MedicalText',
        'ProductReview':'ProductReview',
        'TravelTourism':'TravelTourism',
        'ArtCulture':'ArtCulture',
        'Entertainment':'Entertainment',
        'GovernmentPublic':'GovernmentPublic',
        'NewsArticle':'NewsArticle',
        'Religious':'Religious',
        'Business':'Business',
        'Environmental':'Environmental',
        'LegalDocument':'LegalDocument',
        'OnlineContent':'OnlineContent',
        'Sports':'Sports',
        'Code':'Code',
        'Finance':'Finance',
        'LiteratureCreativeWriting':'LiteratureCreativeWriting',
        'PersonalCommunication':'PersonalCommunication',
        'TechnicalWriting':'TechnicalWriting',
    }
    source_models = {
        'Llama-3-70B': 'Llama-3-70B',
        'GPT-3-Turbo': 'GPT-3-Turbo',
        'Gemini-1.5-Pro': 'Gemini-1.5-Pro',
        # 'GPT-4o': 'GPT-4o',
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        # 'lrr': 'LRR',
        # 'npr': 'NPR', 
        'dna_gpt': 'DNAGPT',
        'binoculars': 'Binoculars',
        'ide_mle': 'IntrinsicDim(MLE)',
        'ide_twonn': 'IntrinsicDim(NN)',
    }
    methods2 = {
        # 'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'revised_gpt2': 'RevisedGPT',
        'rewrite_gpt': 'RewriteGPT(Likelihood)',
        'revised_gpt': 'RewriteGPT(Dist)',
        # 'adarewritegpt': 'RewriteGPT(Ada1)',
        'radar': 'RADAR',
        'raidar': 'RAIDAR',
        'imbd': 'ImBD',
        # 'adarewritegpt_mean_gap': 'RewriteGPT(Ada2)',
        'adarewritegpt_auc': 'RewriteGPT',
        # 'fluoroscopy': 'TextFluoroscopy',
        # 'biscope': 'BiScope',
        # 'classification.bspline': 'AdaDetectGPT',
        # 'superadadetectgpt': 'SuperAdaDetectGPT',
    }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))


def report_topk_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'falcon-7b': 'falcon',}
    methods2 = {
        'sampling_discrepancy': 'FastDetectGPT',
        'variance.identity': 'VarDetectGPT',
        'bivariance.identity': 'BiVarDetectGPT',
        'ml': "MLDetectGPT",
        # 'appro_variance.identity': 'ApproxVarDetectGPT',
        }
    topks = [50, 500, 5000, 0]

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for topk in topks:
            result_file = f'{args.result_path}/{dataset}_falcon-7b_{topk}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        return cols

    def _get_method_stats_gap(dataset, method, filter=''):
        cols = []
        for topk in topks:
            result_file = f'{args.result_path}/{dataset}_falcon-7b_{topk}{filter}.{method}.json'
            if os.path.exists(result_file):
                with open(result_file, 'r') as fin:
                    res = json.load(fin)
                    stats_gap = np.mean(res['predictions']['samples']) - np.mean(res['predictions']['real'])
            else:
                stats_gap = 0.0
            cols.append(stats_gap)
        return cols

    print('>>>>>>>>>>>> Performace >>>>>>>>>>>>')

    headers = ['Top-k'] + [str(top) if top != 0 else '|V|' for top in topks]
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

    print('>>>>>>>>>>>> Gap of Statistics (Fake - Real) >>>>>>>>>>>>')

    headers = ['Top-k'] + [str(top) if top != 0 else '|V|' for top in topks]
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_stats_gap(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))


def report_main_ext_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'bloom-7b1': 'BLOOM-7.1',
                     'opt-13b': 'OPT-13',
                     'llama-13b': 'Llama-13',
                     'llama2-13b': 'Llama2-13',
                     }
    methods1 = {'likelihood': 'Likelihood',
               'entropy': 'Entropy',
               'logrank': 'LogRank',
               'lrr': 'LRR',
               'npr': 'NPR'}
    methods2 = {'perturbation_100': 'DetectGPT',
               'sampling_discrepancy': 'Fast-DetectGPT'}

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        print('(Diff)', ' '.join(cols))
        # black-box comparison
        filters = {'perturbation_100': '.t5-3b_gpt-neo-2.7B',
                    'sampling_discrepancy': '.gpt-j-6B_gpt-neo-2.7B'}
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method, filters[method])
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        print('(Diff)', ' '.join(cols))

def report_refmodel_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J'}

    def _get_method_aurocs(method, ref_model=None):
        cols = []
        for dataset in datasets:
            for model in source_models:
                filter = '' if ref_model is None or ref_model == model else f'.{ref_model}_{model}'
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers1 = ['----'] + list([datasets[d] for d in datasets])
    headers2 = ['Method'] + [source_models[model] for model in source_models] \
              + [source_models[model] for model in source_models] \
              + [source_models[model] for model in source_models] \
              + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))

    ref_models = [None, 'gpt2-xl', 'gpt-neo-2.7B', 'gpt-j-6B']
    for ref_model in ref_models:
        method = 'sampling_discrepancy'
        method_name = 'Fast-DetectGPT (*/*)' if ref_model is None else f'Fast-DetectGPT ({source_models[ref_model]}/*)'
        cols = _get_method_aurocs(method, ref_model)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))

def report_theory_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                    #  'gpt-j-6B': 'GPT-J',
                    #  'gpt-neox-20b': 'NeoX'
                     }
    methods2 = {
        # 'sampling_discrepancy': 'FastDetectGPT(Sampling)', 
        'sampling_discrepancy_analytic': 'FastDetectGPT', 
        # 'classification.identity': 'OracleDetectGPT', 
        # 'classification.bspline': 'AdaDetectGPT(t-test)',
        'classification.bspline_theory': 'AdaDetectGPT(unconstrained)',
        # 'classification.bspline_theory_constrained': 'AdaDetectGPT(constrained)',
        # 'cusumsup': 'SupDetectGPT',
        # 'cusumsup.bspline_theory': 'SupAdaDetectGPT(unconstrained)',
        # 'cusumsup.bspline_theory_constrained': 'SupAdaDetectGPT(constrained)',
        'variance.identity': 'VarDetectGPT',
        'appro_variance.identity': 'ApproxVarDetectGPT',
        }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def report_attack_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                    #  'gpt-j-6B': 'GPT-J',
                    #  'gpt-neox-20b': 'NeoX'
                     }
    methods2 = {
        # 'sampling_discrepancy': 'FastDetectGPT(Sampling)', 
        'sampling_discrepancy_analytic': 'FastDetectGPT', 
        'classification.bspline': 'AdaDetectGPT(t-test)',
        'classification.bspline_theory': 'AdaDetectGPT(unconstrained)',
        'classification.bspline_theory_constrained': 'AdaDetectGPT(constrained)',
        'cusumsup.identity': 'SupDetectGPT',
        'cusumsup.bspline_theory': 'SupAdaDetectGPT(unconstrained)',
        'cusumsup.bspline_theory_constrained': 'SupAdaDetectGPT(constrained)',
        }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}_{args.attack_prop}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))


def report_chatgpt_gpt4_results(args):
    datasets = {
        'xsum': 'XSum',
        'writing': 'Writing',
        # 'pubmed': 'PubMed',
        'yelp': 'Yelp', 
        'essay': 'Essay',
    }
    source_models = {
        # 'gpt-3.5-turbo': 'ChatGPT',
        # 'gpt-4': 'GPT-4',
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini-2.5',
        'claude-3-5-haiku': 'Claude-3.5',
    }
    score_models = { 't5-11b': 'T5-11B',
                     'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX', 
                     'falcon-7b': 'Falcon-7B',
                     'falcon-7b-instruct': 'Falcon-7B-Instruct',
                     'qwen-7b': 'Qwen2.5-7B',
                     'qwen-7b-instruct': 'Qwen2.5-7B-Instruct',
                     'qwen-14b': 'Qwen3-14B',
                     'qwen-14b-base': 'Qwen3-14B-Base',
                     'gemma-9b': 'Gemma2-9B',
                     'gemma-9b-instruct': 'Gemma2-9B-Instruct',
                     }
    methods1 = {
        'roberta-base-openai-detector': 'RoBERTa-base',
        'roberta-large-openai-detector': 'RoBERTa-large'
    }
    methods2 = {
        'likelihood': 'Likelihood', 
        'entropy': 'Entropy', 
        'logrank': 'LogRank',
    }
    methods3 = {
        # 'lrr': 'LRR', 
        # 'npr': 'NPR', 
        # 'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT', 
        'classification.bspline': 'AdaDetectGPT', 
        'supdata.classification.bspline': 'SupAdaDetectGPT', 
        # 'classification': 'AdaDetectGPT',
        # 'classification.debias': 'AdaDetectGPT(Debias)',
    }
    methods4 = {
        'fluoroscopy': 'TextFluoroscopy',
        'radar': 'RADAR',
        'imbd': 'ImBD',
        'biscope': 'BiScope',
        'superadadetectgpt': 'SuperAdaDetectGPT',
    }

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    def _get_method_aurocs2(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    data_list = [datasets[d] for d in datasets] 
    data_list = data_list + ["Avg."] 
    headers2 = ['Method'] + data_list * len(source_models)
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods

    # filters2 = {
    #     'likelihood': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b', '.gemma-9b'],
    #     'entropy': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b', '.gemma-9b'],
    #     'logrank': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b', '.gemma-9b']
    # }
    filters2 = {
        'likelihood': ['.gemma-9b'],
        'entropy': ['.gemma-9b'],
        'logrank': ['.gemma-9b'],
    }
    filters3 = {
        'lrr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'npr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'perturbation_100': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        # 'sampling_discrepancy_analytic': ['.gpt-j-6B_gpt2-xl', '.gpt-j-6B_gpt-neo-2.7B', '.gpt-j-6B_gpt-j-6B', '.gpt-neox-20b_gpt-neox-20b'], 
        # 'classification.bspline': ['.gpt-j-6B_gpt2-xl', '.gpt-j-6B_gpt-neo-2.7B', '.gpt-j-6B_gpt-j-6B', '.gpt-neox-20b_gpt-neox-20b'],
        # 'sampling_discrepancy_analytic': ['.gpt-j-6B_gpt-neo-2.7B', '.falcon-7b_falcon-7b-instruct'], 
        # 'classification.bspline': ['.gpt-j-6B_gpt-neo-2.7B', '.falcon-7b_falcon-7b-instruct'],
        'sampling_discrepancy_analytic': [
            '.falcon-7b_falcon-7b-instruct', 
            ".gemma-9b_gemma-9b-instruct"
        ], 
        'classification.bspline': [
            # '.falcon-7b_falcon-7b-instruct', '.falcon-7b-instruct_falcon-7b', 
            # '.qwen-14b-base_qwen-14b', ".qwen-7b_qwen-7b-instruct", 
            # ".gemma-9b_gemma-9b", ".gemma-9b-instruct_gemma-9b-instruct", 
            ".gemma-9b_gemma-9b-instruct", 
            # ".falcon-7b_falcon-7b-instruct_qwen-7b_qwen-7b-instruct"
        ],
        'supdata.classification.bspline': [
            # '.falcon-7b_falcon-7b-instruct', '.falcon-7b-instruct_falcon-7b', 
            # '.qwen-14b-base_qwen-14b', ".qwen-7b_qwen-7b-instruct", 
            # ".gemma-9b_gemma-9b", ".gemma-9b-instruct_gemma-9b-instruct", 
            ".gemma-9b_gemma-9b-instruct", 
            # ".falcon-7b_falcon-7b-instruct_qwen-7b_qwen-7b-instruct"
        ],
        # 'classification': ['.gpt-j-6B_gpt-j-6B', '.gpt-neo-2.7B_gpt-neo-2.7B',],
        # 'classification.debias': ['.gpt-j-6B_gpt-j-6B', '.gpt-neo-2.7B_gpt-neo-2.7B',],
    }
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods4:
        method_name = methods4[method]
        cols = _get_method_aurocs2(method)
        if method_name == "SuperAdaDetectGPT":
            super_ada_res = cols 
        if method_name == "ImBD":
            imbd_res = cols
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            if methods3[method] == "AdaDetectGPT":
                ada_res = cols 
            if methods3[method] == "SupAdaDetectGPT":
                sup_ada_res = cols
            if method_name == "FastDetectGPT(Gemma2-9B/Gemma2-9B-Instruct)":
                fast_res1 = cols
            if method_name == "FastDetectGPT(Falcon-7B/Falcon-7B-Instruct)":
                fast_res2 = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    # relatives = np.array(ada_res) - np.array(fast_res1)
    # relatives = 100 * relatives / (1.0 - np.array(fast_res1))
    # relatives = [f'{relative:.4f}' for relative in relatives]
    # print('Relative1 (over Fast)', ' '.join(relatives))
    relatives = np.array(ada_res) - np.array(fast_res1)
    relatives = 100 * relatives / (1.0 - np.array(fast_res1))
    relatives = [f'{relative:.4f}' for relative in relatives]
    print('Relative (Ada over Fast)', ' '.join(relatives))
    relatives = np.array(sup_ada_res) - np.array(fast_res1)
    relatives = 100 * relatives / (1.0 - np.array(fast_res1))
    relatives = [f'{relative:.4f}' for relative in relatives]
    print('Relative (SupAda over Fast)', ' '.join(relatives))
    relatives = np.array(super_ada_res) - np.array(fast_res1)
    relatives = 100 * relatives / (1.0 - np.array(fast_res1))
    relatives = [f'{relative:.4f}' for relative in relatives]
    print('Relative (SuperAda over Fast)', ' '.join(relatives))
    relatives = np.array(super_ada_res) - np.array(imbd_res)
    relatives = 100 * relatives / (1.0 - np.array(imbd_res))
    relatives = [f'{relative:.4f}' for relative in relatives]
    print('Relative (over ImBD)', ' '.join(relatives))

def report_gpt3_results(args):
    datasets = {'xsum': 'XSum',
                'writing': 'Writing',
                'pubmed': 'PubMed'}
    source_models = {'davinci': 'GPT-3'}
    score_models = { 't5-11b': 'T5-11B',
                     'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood', 'entropy': 'Entropy', 'logrank': 'LogRank'}
    methods3 = {'lrr': 'LRR', 'npr': 'NPR', 'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'FastDetectGPT', 
                # 'classification.bspline': 'AdaDetectGPT'
                'classification': 'AdaDetectGPT',
                'classification.debias': 'AdaDetectGPT(Debias)',
                }

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    headers2 = ['Method'] + [datasets[dataset] for dataset in datasets] + ['Avg.'] \
               + [datasets[dataset] for dataset in datasets] + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods

    filters2 = {'likelihood': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b'],
               'entropy': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b'],
               'logrank': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b']}
    filters3 = {
        'lrr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'npr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'perturbation_100': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'sampling_discrepancy_analytic': ['.gpt-j-6B_gpt2-xl', '.gpt-j-6B_gpt-neo-2.7B', '.gpt-j-6B_gpt-j-6B', '.gpt-neox-20b_gpt-neox-20b'], 
        'classification': ['.gpt-j-6B_gpt-j-6B', '.gpt-neo-2.7B_gpt-neo-2.7B', ],
        'classification.debias': ['.gpt-j-6B_gpt-j-6B', '.gpt-neo-2.7B_gpt-neo-2.7B',],
    }
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def report_maxlen_trends(args):
    datasets = {'xsum': 'XSum',
                'writing': 'WritingPrompts'}
    source_models = {'gpt-3.5-turbo': 'ChatGPT',
                     'gpt-4': 'GPT-4'}
    score_models = {'t5-11b': 'T5-11B',
                    'gpt2-xl': 'GPT-2',
                    'opt-2.7b': 'OPT-2.7',
                    'gpt-neo-2.7B': 'Neo-2.7',
                    'gpt-j-6B': 'GPT-J',
                    'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood'}
    methods3 = {'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast-Detect'}
    maxlens = [30, 60, 90, 120, 150, 180]

    def _get_method_aurocs(root_path, dataset, source_model, method, filter=''):
        cols = []
        for maxlen in maxlens:
            result_file = f'{root_path}/exp_maxlen{maxlen}/results/{dataset}_{source_model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        return cols

    filters2 = {'likelihood': '.gpt-neo-2.7B'}
    filters3 = {'perturbation_100': '.t5-11b_gpt-neo-2.7B',
                'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-neo-2.7B'}

    headers = ['Method'] + [str(maxlen) for maxlen in maxlens]
    print(' '.join(headers))
    # print table per model and dataset
    results = {}
    for model in source_models:
        model_name = source_models[model]
        for data in datasets:
            data_name = datasets[data]
            print('----')
            print(f'{model_name} / {data_name}')
            print('----')
            for method in methods1:
                method_name = methods1[method]
                cols = _get_method_aurocs('.', data, model, method)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'{col:.4f}' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods2:
                filter = filters2[method]
                setting = score_models[filter[1:]]
                method_name = f'{methods2[method]}({setting})'
                cols = _get_method_aurocs('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'{col:.4f}' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods3:
                filter = filters3[method]
                setting = [score_models[model] for model in filter[1:].split('_')]
                method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
                cols = _get_method_aurocs('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'{col:.4f}' for col in cols]
                print(method_name, ' '.join(cols))
    import json
    json_file = './exp_analysis/maxlen_trends.json'
    with open(json_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Write to file {json_file}')

def report_auroc_curve(args):
    datasets = {'xsum': 'XSum',
                'writing': 'WritingPrompts'}
    source_models = {'gpt-3.5-turbo': 'ChatGPT',
                     'gpt-4': 'GPT-4'}
    score_models = {'t5-11b': 'T5-11B',
                    'gpt2-xl': 'GPT-2',
                    'opt-2.7b': 'OPT-2.7',
                    'gpt-neo-2.7B': 'Neo-2.7',
                    'gpt-j-6B': 'GPT-J',
                    'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood'}
    methods3 = {'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast-Detect'}

    def _get_method_fpr_tpr(root_path, dataset, source_model, method, filter=''):
        maxlen = 180
        result_file = f'{root_path}/exp_maxlen{maxlen}/results/{dataset}_{source_model}{filter}.{method}.json'
        if os.path.exists(result_file):
            fpr, tpr = get_fpr_tpr(result_file)
        else:
            fpr, tpr = [], []
        assert len(fpr) == len(tpr)
        return list(zip(fpr, tpr))

    filters2 = {'likelihood': '.gpt-neo-2.7B'}
    filters3 = {'perturbation_100': '.t5-11b_gpt-neo-2.7B',
                'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-neo-2.7B'}

    # print table per model and dataset
    results = {}
    for model in source_models:
        model_name = source_models[model]
        for data in datasets:
            data_name = datasets[data]
            print('----')
            print(f'{model_name} / {data_name}')
            print('----')
            for method in methods1:
                method_name = methods1[method]
                cols = _get_method_fpr_tpr('.', data, model, method)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'({col[0]:.3f},{col[1]:.3f})' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods2:
                filter = filters2[method]
                setting = score_models[filter[1:]]
                method_name = f'{methods2[method]}({setting})'
                cols = _get_method_fpr_tpr('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'({col[0]:.3f},{col[1]:.3f})' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods3:
                filter = filters3[method]
                setting = [score_models[model] for model in filter[1:].split('_')]
                method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
                cols = _get_method_fpr_tpr('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'({col[0]:.3f},{col[1]:.3f})' for col in cols]
                print(method_name, ' '.join(cols))
    import json
    json_file = './exp_analysis/auroc_curve.json'
    with open(json_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Write to file {json_file}')

def report_multi_auroc_curve(arg):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX'}
    methods1 = {'likelihood': 'Likelihood',
               'entropy': 'Entropy',
               'logrank': 'LogRank',
               'lrr': 'LRR',
               'npr': 'NPR', 
               'dna_gpt': 'DNAGPT',}
    methods2 = {
        'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT', 
        'classification.bspline_theory': 'AdaDetectGPT',
        'classification.multi.bspline_theory': 'MultiAdaDetectGPT',
        }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def get_typeIerror(result_file, alpha=0.1):
    critical_value = norm.ppf(1 - alpha, loc=0.0, scale=1.0)
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        return np.mean(real_stats > critical_value)

def get_power(result_file, alpha=0.1):
    critical_value = norm.ppf(1 - alpha, loc=0.0, scale=1.0)
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        fake_stats = np.array(res['predictions']['samples'])
        return np.mean(fake_stats > critical_value)  

def get_tpr(result_file, alpha=0.1):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        fake_stats = np.array(res['predictions']['samples'])
        critical_value = np.quantile(fake_stats, q=alpha)
        return np.mean(real_stats <= critical_value)

def get_fpr(result_file, alpha=0.1):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        fake_stats = np.array(res['predictions']['samples'])
        critical_value = np.quantile(real_stats, q=1-alpha)
        return np.mean(fake_stats <= critical_value)  

def report_typeIerror_power(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                    #  'gpt-j-6B': 'GPT-J',
                    #  'gpt-neox-20b': 'NeoX', 
                     }
    methods2 = {
        'bspline.debias': 'bspline(debias)',
        'bspline': 'bspline',
        'identity.debias': 'identity(debias)',
        'identity': 'identity',
        }

    def _get_method_typeIerror(dataset, method, alpha):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}.classification.{method}.json'
            if os.path.exists(result_file):
                error = get_typeIerror(result_file, alpha)
            else:
                error = -9999.99
            cols.append(error)
        cols.append(np.mean(cols))
        return cols

    def _get_method_power(dataset, method, alpha):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}.classification.{method}.json'
            if os.path.exists(result_file):
                error = get_power(result_file, alpha)
            else:
                error = -9999.99
            cols.append(error)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    # white-box comparison
    for dataset in datasets:
        print('----')
        print(datasets[dataset] + ' Type I Error')
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_typeIerror(dataset, method, args.alpha)
            results[method_name] = cols
            cols = [f'{col:.3f}' for col in cols]
            print(method_name, ' '.join(cols))
    for dataset in datasets:
        print('----')
        print(datasets[dataset] + ' Power')
        print('----')
        print(' '.join(headers))
        results_power = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_power(dataset, method, args.alpha)
            results_power[method_name] = cols
            cols = [f'{col:.3f}' for col in cols]
            print(method_name, ' '.join(cols))

        ## black-box comparison
        # filters = {'perturbation_100': '.t5-3b_gpt-neo-2.7B',
        #             'sampling_discrepancy': '.gpt-j-6B_gpt-neo-2.7B'}
        # results = {}
        # for method in methods2:
        #     method_name = methods2[method]
        #     cols = _get_method_aurocs(dataset, method, filters[method])
        #     results[method_name] = cols
        #     cols = [f'{col:.4f}' for col in cols]
        #     print(method_name, ' '.join(cols))
        # cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        # cols = [f'{col:.4f}' for col in cols]
        # print('(Diff)', ' '.join(cols))

def report_tpr_fpr(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                    #  'gpt-j-6B': 'GPT-J',
                    #  'gpt-neox-20b': 'NeoX', 
                     }
    methods2 = {
        'classification': 'AdaDetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        }

    def _get_method_fpr(dataset, method, alpha):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}.{method}.json'
            if os.path.exists(result_file):
                error = get_fpr(result_file, alpha)
            else:
                error = 0.0
            cols.append(error)
        cols.append(np.mean(cols))
        return cols

    def _get_method_tpr(dataset, method, alpha):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}.{method}.json'
            if os.path.exists(result_file):
                error = get_tpr(result_file, alpha)
            else:
                error = 0.0
            cols.append(error)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    # white-box comparison
    for dataset in datasets:
        print('----')
        print(datasets[dataset] + ' (FPR)')
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_fpr(dataset, method, args.alpha)
            results[method_name] = cols
            cols = [f'{col:.3f}' for col in cols]
            print(method_name, ' '.join(cols))
    for dataset in datasets:
        print('----')
        print(datasets[dataset] + ' (TPR)')
        print('----')
        print(' '.join(headers))
        results_power = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_tpr(dataset, method, args.alpha)
            results_power[method_name] = cols
            cols = [f'{col:.3f}' for col in cols]
            print(method_name, ' '.join(cols))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--report_name', type=str, default="prompt_results")
    # parser.add_argument('--report_name', type=str, default="black_prompt_results")
    parser.add_argument('--result_path', type=str, default="./exp_diverse/results/")
    parser.add_argument('--report_name', type=str, default="diverse_results")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/paraphrasing")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/decoherence")
    # parser.add_argument('--result_path', type=str, default="./exp_attack/results/")
    # parser.add_argument('--report_name', type=str, default="attack_results")
    # parser.add_argument('--alpha', type=float, default=0.10)
    # parser.add_argument('--attack_prop', type=float, default=0.05)
    parser.add_argument('--attack_prop', type=int, default=150)
    args = parser.parse_args()

    if args.report_name == 'black_prompt_results':
        report_black_prompt_results(args)
    if args.report_name == 'diverse_results':
        report_diverse_results(args)