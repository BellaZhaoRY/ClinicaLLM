import re

import json

import os
from config.config import *
import pandas as pd
# from transformers import AutoTokenizer, AutoModel
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# my_model_path = os.path.join(model_dir_path,"")
# tokenizer = AutoTokenizer.from_pretrained(model_dir_path+"/chatglm2-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained(model_dir_path, trust_remote_code=True).cuda()

# tokenizer = AutoTokenizer.from_pretrained("THUDM/ChatGLM2-6B", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/ChatGLM2-6B", trust_remote_code=True,cache_dir=model_dir_path).cuda()
# r, h = model.chat(tokenizer, query, history=h, do_sample=False)

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="THUDM/chatglm-6b", local_dir=model_dir_path+"/chatglm-6b/")

from modelscope.utils.constant import Tasks  
from modelscope import Model
from modelscope.pipelines import pipeline
# DEVICE = "cuda"
# DEVICE_ID = ""
# CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map='auto', revision='v1.0.12')
# model = Model.from_pretrained('ZhipuAI/chatglm2-6b', revision='v1.0.7',max_length=31*1024,do_sample=False,device_map='auto')
# model = Model.from_pretrained('ZhipuAI/chatglm2-6b', device_map=CUDA_DEVICE, revision='v1.0.7',max_length=31*1024,do_sample=False)

pipe = pipeline(task=Tasks.chat, model=model,topk=0.000000001,temperature=0.000000001)
inputs = {"text": prompt, "history": history}
result = pipe(inputs)

def get_result_4_re(rule_info, context_4_stem_vid):
    # 基于正则的方式获取结果
    res_dict = {}
    res = re.finditer(rule_info, context_4_stem_vid)
    if res:
        for x in res:
            for group_name,group_value in  x.groupdict():
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
    :param rule_info: stem中补充的医学知识
    :param context_4_stem_vid: 该就诊中关于该stem的相关病历文本信息
    :return:
    """
    prompt = f"你将获得一份在{file_re_info}中关于{sec_index_re}的相关病历，请你根据‘’‘{rule_info}’‘’中的医学知识获得‘’‘{stem_cn_name}’‘’问题的答案，\n病历：{context_4_stem_vid}。因此{stem_cn_name}的答案为"
    inputs = {"text": prompt, "history": []}
    result = pipe(inputs)
    return result
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
    all_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items()]))
    rule_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if rl == "规则"]))
    # model_res = list(set([ans for xdict in stem_results for rl,ans in xdict.items() if rl == "模型"]))

    if len(all_res) == 1:
        return all_res[0]
    else:
        if len(rule_res) == 1:
            return rule_res[0]
        else:
            if "y" in rule_res:
                return "y"
            else:
                return "n"
def get_context_info_4_vid_4_stem(file_re_info,sec_index_re,cli_info_4_vid):
    context = ""
    context_format = "该病历%s中关于%s的信息为：%s"
    for file_name, cli_info in cli_info_4_vid.items():
        if re.search(file_re_info, file_name):
            for sec_index, sec_info in cli_info.items():
                if re.search(sec_index_re, sec_index):
                    context += context_format % (file_name, sec_index, sec_info)
    return context
def get_all_stem_info():
    result_dict_path = os.path.join(prepro_orig_data_dir_path, "result_dict.json")
    with open(result_dict_path,"r",encoding="utf-8") as f:
        stem_info_dict = json.load(f)
    new_stem_info_dict = {}
    for stem_name,stem_info in stem_info_dict.items():
        if not re.search("STEMI-[456]|STEMI-3-2-7",stem_name):  # todo
            continue
        else:
            new_stem_info_dict[stem_name] = stem_info
    return new_stem_info_dict
def get_check_vids_info():
    # 读取就诊列表中的就诊id信息，并核对和解析的数据中vid是否相同
    with open(os.path.join(orig_data_dir_path,"3-就诊流水号列表.txt"),"r",encoding="utf-8") as f:
        all_vids = f.readlines()
    all_vids = [x.strip() for x in all_vids if x]
    prepro_vids = os.listdir(prepro_data_dir_path)
    assert len(all_vids) == len(prepro_vids)
    return all_vids
def get_cli_info_4_vid(files_4_vid):
    # 读取该就诊下的所有病历信息
    cli_info_4_vid = {}
    for file_4_cli_info in files_4_vid:
        with open(os.path.join(prepro_data_dir_path, file_4_cli_info), "r", encoding="utf-8") as f:
            cli_info = json.load(f)
        cli_info_4_vid[file_4_cli_info[:-5]] = cli_info
    return cli_info_4_vid
def main():
    # 1. 读取stem的配置信息
    new_stem_info_dict = get_all_stem_info()
    # 2. 读取就诊流水号信息
    all_vids = get_check_vids_info()
    # 3. 生产每个就诊的每个stem问题结果
    for vid in all_vids:
        vid_file_path = os.path.join(prepro_data_dir_path,vid)
        files_4_vid = os.listdir(vid_file_path)
        # 3.1 加载该就诊下的所有转化格式后的病历信息
        cli_info_4_vid = get_cli_info_4_vid(files_4_vid)
        # 3.2 依次获得每个stem的结果，需要根据有向无环图的顺序获取结果
        for stem_name,stem_info in new_stem_info_dict.items():
            stem_cn_name = stem_info["数据采集项"]
            stem_type = stem_info["数据类型"]
            stem_other_info = stem_info["备注"]
            stem_select_info = stem_info["选项列表"]
            stem_rule_info = stem_info["规则信息"]
            stem_results = []
            # 3.3 通过stem信息获得结果
            for stem_info in stem_rule_info:
                if isinstance(stem_info,dict):
                    file_re_info = stem_info["文件名"].strip()
                    sec_index_re = stem_info["表索引"].strip()
                    parser_fun = stem_info["解析方式"].strip()
                    rule_info = stem_info["规则"].strip()
                    # 3.4 获取该就诊关于该stem的相关病历内容。
                    context_4_stem_vid = get_context_info_4_vid_4_stem(file_re_info, sec_index_re, cli_info_4_vid)
                    # 3.5 根据规则或模型，获取相应答案，并使答案符合比赛要求
                    if parser_fun == "规则":
                        stem_res_4_rule = get_result_4_re(rule_info,context_4_stem_vid)
                        stem_results.append({"规则":stem_res_4_rule})
                    elif parser_fun == "模型":
                        stem_res_4_model = get_result_4_model(file_re_info, sec_index_re,stem_cn_name,rule_info, context_4_stem_vid)
                        stem_results.append({"模型":stem_res_4_model})
                    else:
                        raise print(f"{stem_name}\t{stem_cn_name}\t的解析方式为{parser_fun},错误.")
                else:
                    raise print(f"{vid}就诊中{stem_name}:{stem_cn_name}的stem_info应该解析为dict，但是实际为{stem_info}，错误！")
                # 3.6 观察发现有多条结果的是”禁忌症“相关stem字段，答案只有y，n，合并答案
                stem_res = combine_stem_res_4_rule_and_model(stem_results)
                try:
                    # 3.7 数据类型为字符串的答案结果应该只有一个，因此需要核对结果
                    check_stem_res_and_type(stem_type,stem_res)
                except:
                    raise print(f"{vid}就诊中{stem_name}:{stem_cn_name}的数据类型为{stem_type}，答案为{stem_res},不符合要求。")

if __name__ == '__main__':
    main()
