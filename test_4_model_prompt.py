import re

import json

import os
from config.config import *
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# my_model_path = "/home/ubuntu/hyx/LLaMA-Efficient-Tuning-main/HYX_ChatGLM2_"
# tokenizer = AutoTokenizer.from_pretrained(my_model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(my_model_path, trust_remote_code=True).cuda()

# tokenizer = AutoTokenizer.from_pretrained("THUDM/ChatGLM2-6B", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/ChatGLM2-6B", trust_remote_code=True,cache_dir=model_dir_path).cuda()
# r, h = model.chat(tokenizer, query, history=h, do_sample=False)

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="THUDM/chatglm-6b", local_dir=model_dir_path+"/chatglm-6b/")

# from modelscope.utils.constant import Tasks  
# from modelscope import Model
# from modelscope.pipelines import pipeline
# DEVICE = "cuda"
# DEVICE_ID = ""
# CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
# model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map='auto', revision='v1.0.12')
# model = Model.from_pretrained('ZhipuAI/chatglm2-6b', revision='v1.0.7',max_length=31*1024,do_sample=False,device_map='auto')
# model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map=CUDA_DEVICE, revision='v1.0.7',max_length=31*1024,do_sample=False)

# pipe = pipeline(task=Tasks.chat, model=model,topk=0.000000001,temperature=0.000000001)
# inputs = {"text": prompt, "history": history}
# result = pipe(inputs)

# 有向无环图中各结点的入度，以及每个结点指向的所有结点
def prepare_degs_and_tails(dependencies):
    degs = dict()
    tails = dict((k, []) for k in dependencies)
    for k, v in dependencies.items():
        degs[k] = len(v)
        for item in v:
            tails[item].append(k)
    return degs, tails
# 拓扑排序
def toposort(degs, tails):
    res = []
    q = set()
    for term, in_deg in degs.items():
        if in_deg == 0:
            q.add(term)
    while len(q) > 0:
        front = q.pop()
        res.append(front)
        for term in tails[front]:
            degs[term] -= 1
            if degs[term] == 0:
                q.add(term)
    return res

# 根据所依赖数据项的预测值判断是否需要填写当前数据项
def check_whether_ans_via_dependency(term, dependent_terms, stem_2_answer):
    flag_must_ans = False
    flag_may_ans = True
    flag_need_not_ans = False
    for dependent_term in dependent_terms:
        depen_term_ans =  stem_2_answer[dependent_term]
        if (term, dependent_term) in must_answer and depen_term_ans == must_answer[(term, dependent_term)]:
            flag_must_ans = True
        if (term, dependent_term) in may_answer and depen_term_ans != may_answer[(term, dependent_term)]:
            flag_may_ans = False
        if (term, dependent_term) in need_not_answer and depen_term_ans == need_not_answer[(term, dependent_term)]:
            flag_need_not_ans = True
    return flag_must_ans, flag_may_ans, flag_need_not_ans
# 根据有向无环图进行后处理
def post_processing(vid_2_stem_answer):
    new_vid_2_stem_answer = deepcopy(vid_2_stem_answer)
    for vid, stem_2_answer in vid_2_stem_answer.items():
        for stem, ans in stem_2_answer.items():
            # 该数据项所依赖的所有数据项
            depen_terms = term_dependencies[stem]
            must_ans, may_ans, need_not_ans = check_whether_ans_via_dependency(stem, depen_terms, stem_2_answer)
            # 必须填写
            if must_ans:
                assert need_not_ans == False
                if not ans:
                    print(f"[{vid}][Conflict][Must answer]:"
                          f"{stem}无预测值，与其依赖项的预测结果{[(x, stem_2_answer[x]) for x in depen_terms]}矛盾")
            # 不需要填写
            if not may_ans or need_not_ans:
                if ans:
                    print(f"[{vid}][Conflict][Need not answer]:"
                          f"{stem}有预测值{ans}，与其依赖项的预测结果{[(x, stem_2_answer[x]) for x in depen_terms]}矛盾")
                new_vid_2_stem_answer[vid][stem] = None
    return new_vid_2_stem_answer

def get_result_4_re(rule_info, context_4_stem_vid):
    # 基于正则的方式获取结果
    res_dict = {}
    # print(rule_info)
    res = re.finditer(rule_info, context_4_stem_vid)
    if res:
        for x in res:
            for group_name,group_value in  x.groupdict().items():
                if group_value:
                    res_dict[group_name] = group_value
    else:
        return ""
    res_group_name = [re.sub("\d+","",x) for x in list(res_dict.keys()) if x]
    res_group_name = sorted(list(set(res_group_name)))
    return "\\".join(res_group_name)
def get_result_4_model(file_re_info, sec_index_re,stem_cn_name,rule_info,context_4_stem_vid):
    """
    根据对现有27-48中的stem的规则发现，只有三个关于”禁忌症“的字段需要模型进行问答，以此作为特征构建prompt。
    :param file_re_info: 文件名信息，即一级索引
    :param sec_index_re: 表索引信息，即二级信息
    :param stem_cn_name: stem中文名
    :param rule_info: prompt
    :param context_4_stem_vid: 该就诊中关于该stem的相关病历文本信息
    :return:
    """
    rule_info = rule_info.replace("\\n", "\n")
    prompt = f"你将阅读一段来自{file_re_info}的病历文本，并根据病历内容回答一个问题。\n病历文本：\n{context_4_stem_vid}\n根据病历内容，请问{rule_info}"
    r, h = model.chat(tokenizer, prompt, history=[], do_sample=False)
    # print("*"*100)
    # print(prompt)
    # print()
    # print(r)
    # print("*"*100)
    result = parse_model_answer(r, stem_cn_name)
    return result
def parse_model_answer(response, term):
    # 对模型回复进行解析得到答案
    # if response in ['是', '否']:
    #     return yes_or_no_mapping[response]

    res_dict = {}
    mapping_rule = "(?P<y>是)|(?P<n>否)|(?P<a>A\.)|(?P<b>B\.)|(?P<c>C\.)|(?P<d>D\.)|(?P<UTD>E\.)"
    res = re.finditer(mapping_rule, response)
    if res:
        for x in res:
            for group_name,group_value in  x.groupdict().items():
                if group_value:
                    res_dict[group_name] = group_value
    else:
        print(response)
        exit(0)
    res_group_name = [re.sub("\d+","",x) for x in list(res_dict.keys()) if x]
    res_group_name = sorted(list(set(res_group_name)))
    return "\\".join(res_group_name)
def check_stem_res_and_type(stem_type, stem_res):
    if stem_type == "字符串":
        assert len(stem_res.split("\\")) == 1
    elif stem_type == "数组":
        pass
    else:
        raise print("数据采集项的数据类型既不是字符串也不是数组，不符合要求！")
def combine_stem_res_4_rule_and_model(stem_results):
    # stem_results：[{规则：答案},{模型：答案}]
    # 若既有规则又有模型，获取答案的依据是以规则为准==》若规则答案冲突，则以y为准 todo
    all_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if ans]))
    rule_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if rl == "规则" and ans]))
    model_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if rl == "模型" and ans]))

    if len(all_res) == 1:
        return all_res[0]
    elif len(rule_res) == 1:
        return rule_res[0]
    elif len(model_res) == 1:
        return model_res[0]
    else:
        if "y" in rule_res:
            return "y"
        else:
            return "n"

def get_precondition_2_secect_line(stem_cn_name,condition_info):
    """

    :param stem_cn_name:
    :param condition_info:
    :return:
    """
    if re.search("首",stem_cn_name):
        return condition_info.get("首次入院时间","")
    elif re.search("围术期",stem_cn_name):
        terms = condition_info.get("围术期",[])
        return terms
        # for term in terms:
        #     for start_time,end_time in term:
        #         start_time = int(start_time) -1
        #         end_time = int(end_time) +1
        #         return
    else:
        return ""
def is_filter_4_sec_index(precondition_re,sec_index):
    if isinstance(precondition_re, str):
        if re.search(precondition_re, sec_index):
            return False
    else:
        for pre_re in precondition_re:
            for start_time, end_time in pre_re:
                if start_time <= int(sec_index[-2]) <= end_time:
                    return False
    return True
def get_context_info_4_vid_4_stem(file_re_info,sec_index_re,cli_info_4_vid,line_re,stem_cn_name):
    # stem_cn_name: 如果是“首”相关，则医嘱规则增加 首次入院时间和sec_index中医嘱开始相同相同；
    #                 如果是“围术期”,则医嘱规则增加 医嘱开始和结束时间限制下 sec_index筛选条件
    context = ""
    # 获取前置条件
    precondition_re = get_precondition_2_secect_line(stem_cn_name,cli_info_4_vid.get("补充信息",{}))
    for file_name, cli_info in cli_info_4_vid.items():
        if re.search(file_re_info, file_name):
            for sec_index, sec_info in cli_info.items():
                if precondition_re and re.search("医嘱",file_re_info):
                    if is_filter_4_sec_index(precondition_re, sec_index):
                       continue
                sec_index = re.sub("_\d{4,}","",sec_index)  # 删除掉时间信息
                if re.search(sec_index_re, sec_index):
                    # 通过line_re行筛选依据，过滤无用信息
                    if isinstance(sec_info,list):
                        line_str = "\n".join([x for x in sec_info if re.search(line_re,x)])
                    else:
                        line_str = sec_info if re.search(line_re,sec_info) else ""
                    context += line_str + '\n'
    return context
def get_all_stem_info():
    result_dict_path = os.path.join(prepro_orig_data_dir_path, "result_dict.json")
    with open(result_dict_path,"r",encoding="utf-8") as f:
        stem_info_dict = json.load(f)
    # 有向无环图
    degs, tails = prepare_degs_and_tails(term_dependencies)
    # 拓扑排序
    term_seq = toposort(degs, tails)
    assert len(term_seq) == len(stem_info_dict)
    new_stem_info_dict = {}
    for stem_name in term_seq:
        new_stem_info_dict[stem_name] = stem_info_dict[stem_name]
    return new_stem_info_dict
def get_check_vids_info():
    # 读取就诊列表中的就诊id信息，并核对和解析的数据中vid是否相同
    with open(os.path.join(orig_data_dir_path,"3-就诊流水号列表.txt"),"r",encoding="utf-8") as f:
        all_vids = f.readlines()
    all_vids = [x.strip() for x in all_vids if x]
    prepro_vids = os.listdir(prepro_data_dir_path)
    assert len(all_vids) == len(prepro_vids)
    return all_vids
def get_cli_info_4_vid(vid_file_path,files_4_vid):
    # 读取该就诊下的所有病历信息
    cli_info_4_vid = {}
    for file_4_cli_info in files_4_vid:
        with open(os.path.join(vid_file_path, file_4_cli_info), "r", encoding="utf-8") as f:
            cli_info = json.load(f)
        cli_info_4_vid[file_4_cli_info[:-5]] = cli_info
    return cli_info_4_vid

def covert_dict_2_pd(vid_2_stem_answer):
    new_res = []
    for vid,stem_info in vid_2_stem_answer.items():
        for name,res in stem_info.items():
            new_res.append({"就诊流水号":vid,	"填报数据项编码":name,	"选项或数据值":res})
    return new_res

def compare_results(vid_2_stem_answer, gold_annotaion_path):
    compare_res,stem_names_pre,stem_names_gold = [],set(),set()
    vid_2_stem_gold =pd.read_excel(gold_annotaion_path,usecols=["就诊流水号", "填报数据项编码", "选项或数据值"]).fillna("").astype(str)
    vid_2_stem_gold.set_index(["就诊流水号", "填报数据项编码"],inplace = True)
    new_vid_2_stem_gold = defaultdict(dict)
    for (vid,name),res_gold in list(vid_2_stem_gold.to_dict().values())[0].items():
        new_vid_2_stem_gold[vid].update({name:re.sub("^\"|\"$","",res_gold)})
        stem_names_gold.update({name})
    eq_num = 0
    for vid,stem_info_pre_4_vid in vid_2_stem_answer.items():
        stem_info_gold_4_vid = new_vid_2_stem_gold.get(vid,{})
        for name,value_pred in stem_info_pre_4_vid.items():
            stem_names_pre.update({name})
            value_gold = stem_info_gold_4_vid.get(name,"")
            is_equal = 1 if value_pred == value_gold else ""
            eq_num +=1 if is_equal else 0
            compare_res.append({"就诊流水号":vid, "填报数据项编码":name, "选项或数据值_pre":value_pred,"选项或数据值_gold":value_gold,"是否正确":is_equal})
    pre_res_all = [x.get("选项或数据值_pre") for x in compare_res if x.get("选项或数据值_pre")]
    gold_res_all = [x.get("选项或数据值_gold") for x in compare_res if x.get("选项或数据值_gold")]
    # print("pre_res_all",pre_res_all)
    # print("gold_res_all",gold_res_all)
    # print(stem_names_pre)
    # print(stem_names_gold)
    print(f"pre的数据采集项数量为{len(stem_names_pre)}，比比赛多的为{stem_names_pre-stem_names_gold}；gold的数据数据采集项数量为{len(stem_names_gold)}，比48个多的为{stem_names_gold-stem_names_pre}！")
    print(f"pre非空结果数量{len(pre_res_all)}，gold非空结果数量{len(gold_res_all)}。准确率：{eq_num}/{len(compare_res)}，{eq_num/len(compare_res) * 100:.2f}%  !")
    pd.DataFrame.from_records(compare_res).to_excel(os.path.join(results_dir_path,"结果对比.xlsx"))

def main():
    # 1. 读取stem的配置信息
    new_stem_info_dict = get_all_stem_info()
    # 2. 读取就诊流水号信息
    all_vids = get_check_vids_info()
    # 对于每个数据项，各个就诊流水号对应的预测答案
    vid_2_stem_answer = dict((vid, {}) for vid in all_vids)
    # 3. 生产每个就诊的每个stem问题结果
    for vid in all_vids:
        vid_file_path = os.path.join(prepro_data_dir_path,vid)
        files_4_vid = os.listdir(vid_file_path)
        # 3.1 加载该就诊下的所有转化格式后的病历信息
        cli_info_4_vid = get_cli_info_4_vid(vid_file_path,files_4_vid)
        # 3.2 依次获得每个stem的结果，需要根据有向无环图的顺序获取结果
        for stem_name,stem_info in new_stem_info_dict.items():
            stem_cn_name = stem_info["数据采集项"]
            stem_type = stem_info["数据类型"]
            stem_other_info = stem_info["备注"]
            stem_select_info = stem_info["选项列表"]
            stem_rule_info = stem_info["规则信息"]
            stem_results = []

            # 对于某些数据项，可以根据其依赖的数据项的预测值直接得到答案（例如STEMI-3-2-1为y则STEMI-3-2-2为n）
            if stem_name in infer_answer_via_dependency:
                for depen_stem, depen_ans_mapping in infer_answer_via_dependency[stem_name].items():
                    assert depen_stem in vid_2_stem_answer[vid]
                    depen_stem_answer = vid_2_stem_answer[vid][depen_stem]
                    if depen_stem_answer in depen_ans_mapping:
                        inferred_stem_res = depen_ans_mapping[depen_stem_answer]
                        # print(f"inferred_stem_res: {inferred_stem_res}")
                        stem_results.append({"规则": inferred_stem_res})
            
            # 3.3 通过stem信息获得结果
            for stem_info in stem_rule_info:
                if isinstance(stem_info,dict):
                    file_re_info = stem_info["文件名"].strip()
                    sec_index_re = stem_info["表索引"].strip()
                    line_re = stem_info["行筛选条件"].strip()
                    parser_fun = stem_info["解析方式"].strip()
                    rule_info = stem_info["规则"].strip()
                    # 3.4 获取该就诊关于该stem的相关病历内容。
                    context_4_stem_vid = get_context_info_4_vid_4_stem(file_re_info, sec_index_re, cli_info_4_vid,line_re,stem_cn_name)
                    # 3.5 根据规则或模型，获取相应答案，并使答案符合比赛要求
                    if parser_fun == "规则":
                        stem_res_4_rule = get_result_4_re(rule_info,context_4_stem_vid)
                        stem_results.append({"规则":stem_res_4_rule})
                    elif parser_fun == "模型":
                        # stem_res_4_model = get_result_4_model(file_re_info, sec_index_re,stem_cn_name,rule_info, context_4_stem_vid)
                        # stem_results.append({"模型":stem_res_4_model})
                        # print(f"模型预测答案为{stem_res_4_model}\n")
                        pass
                    else:
                        raise print(f"{stem_name}\t{stem_cn_name}\t的解析方式为{parser_fun},错误.")
                else:
                    raise print(f"{vid}就诊中{stem_name}:{stem_cn_name}的stem_info应该解析为dict，但是实际为{stem_info}，错误！")
            # 3.6 观察发现有多条结果的是”禁忌症“相关stem字段，答案只有y，n，合并答案
            stem_res = combine_stem_res_4_rule_and_model(stem_results)
            vid_2_stem_answer[vid][stem_name] = stem_res
            try:
                # 3.7 数据类型为字符串的答案结果应该只有一个，因此需要核对结果
                check_stem_res_and_type(stem_type,stem_res)
            except:
                # print(f"{vid}就诊中{stem_name}:{stem_cn_name}的数据类型为{stem_type}，规则为\n{stem_rule_info}\n预测答案为{stem_res}，不符合要求。\n")
                pass
    # 3.8 根据有向无环图进行后处理
    vid_2_stem_answer = post_processing(vid_2_stem_answer)

    # 3.9 保存excel格式的模型预测结果
    vid_2_stem_answer_4_pd = covert_dict_2_pd(vid_2_stem_answer)
    pd.DataFrame(vid_2_stem_answer_4_pd).to_excel(os.path.join(results_dir_path, "预测结果.xlsx"),)


    # 4. 将模型预测结果和标注结果对比
    gold_annotaion_path = "data/orig_datas/8-填报结果.xlsx"
    if os.path.exists(gold_annotaion_path):
        compare_results(vid_2_stem_answer, gold_annotaion_path)


if __name__ == '__main__':
    main()
