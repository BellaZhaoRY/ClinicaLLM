import os.path
import re

# 1. 文件

config_dir_path = os.path.abspath(os.path.dirname(__file__))
program_dir_path = os.path.join(config_dir_path, os.path.pardir)
# 以下是data目录
data_dir_path = os.path.join(program_dir_path, 'data')
orig_data_dir_path = os.path.join(data_dir_path, 'orig_datas')
clinical_data_dir_path = os.path.join(orig_data_dir_path, '4-病历文书')
stem_file = "STEMI数据项说明_备注说明_补充信息_v2.xlsx"
prepro_data_dir_path = os.path.join(data_dir_path, 'prepro_data')
prepro_orig_data_dir_path = os.path.join(data_dir_path, 'prepro_orig_datas')
results_dir_path = os.path.join(program_dir_path, 'results')
# 以下是model目录
model_dir_path = os.path.join(program_dir_path,"model")


# 2. doctype
doc_type_dict={"入院记录":"admit_note",
               "出院记录":"discharge_note",
               "冠状动脉造影":"pci_note", # 急诊冠状动脉造影及PCI记录,择期冠状动脉造影记录,择期冠状动脉造影及PCI记录

               "超声心动图结果":"ultrasound_result",
               "医嘱":"medical_order",
               "其他记录":"other_notes",
                "检验报告":"lab_report"}
# second_table_dir_dict 根据正则获取核心词，然后根据核心词获取二级表名
# 中文的正则表达式是
sc_table_dict_re = {"admit_note":"^\S\s\S\s\S\s\S|^[\u4e00-\u9fa5 ]*?[：:]\s*$",
                     "discharge_note":"^[一二三四五六七八九十]*、[\u4e00-\u9fa5 ]+?[:：]",
                     "pci_note": "(择期冠状动脉造影及PCI记录|急诊冠状动脉造影及PCI记录|择期冠状动脉造影记录|结论|备注|介入治疗结果|介入治疗基本信息|介入时间|术后医嘱|并发症|辅助设备|其他影像学|术后安返病房)"}
# 无意义词列表 正则re
stop_words = ['的', '是', '在', '我', '你',"否","有","无","\*+|[a-z][: ]|def|请|是否|有无|[\n\s\t\\t\\n]+|[、，。；\?]+","使用","的","选择",":",";"]

# 备注3中的信息进行解析
# rule_2_parser = "(?P<first_layer>.*?)\.(?P<sec_layer>.*?)【(?P<re_model>.*?)】[:：](?P<info>.*)"
# rule_2_ls_parser = "[;；]"

