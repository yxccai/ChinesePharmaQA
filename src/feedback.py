from methods.llm_parser import FIELD_DEFINITIONS
PROMPT='''你是药典集合查询的校验反馈修复器。你会收到原始问题、模型首次预测的约束、字段定义，以及程序给出的逐项校验结果。
校验规则是字段与规范名称的精确匹配。INVALID_CANONICAL表示字符串不属于该字段的规范值，不等于原问题的条件不存在，也不等于条件的语义一定错误。
任务：只修复列在failed_constraints中的约束。根据原问题和校验反馈重新判断术语的规范表达，输出该字段的简洁名称，而不是来源说明、完整叙述或操作描述。不得改变条件的含义和范围；不得改变字段、删除条件、合并条件或添加新条件。已通过校验的约束由程序原样保留，无需输出。
本次不提供检索证据或合法值菜单。请利用已有知识和问题语境直接尝试修复，不请求检索；无法确定时返回UNRESOLVED。不要将原封不动的无效字符串当成成功修复。
仅输出JSON：{"repairs":[{"index":原约束索引,"decision":"ACCEPT","field":"原字段","canonical":"修复后的规范名称"}]}。
无法修复的项目输出{"index":原约束索引,"decision":"UNRESOLVED"}。必须逐项覆盖所有失败索引，不输出药品名称、解释、候选列表或其他字段。'''

def execute(cs,exact,post):
    ids=[exact.get((c.get('field'),c.get('canonical'))) for c in cs]
    return sorted(set.intersection(*(post[i] for i in ids))) if ids and all(ids) and len(set(ids))==len(ids) else []

def validate_output(value,payload,exact,post):
    initial=payload['initial_constraints'];cs=[dict(c) for c in initial];failed={x['index']:x for x in payload['failed_constraints']}
    repairs=value.get('repairs');errors=[]
    if not isinstance(repairs,list) or any(not isinstance(x,dict) for x in repairs):return cs,[],['MALFORMED_REPAIRS']
    idx=[x.get('index') for x in repairs]
    if any(type(i)!=int for i in idx) or len(set(idx))!=len(idx) or set(idx)!=set(failed):return cs,[],['INVALID_REPAIR_INDICES']
    for r in repairs:
        i=r['index']
        if r.get('decision')=='UNRESOLVED':continue
        if r.get('decision')!='ACCEPT' or r.get('field')!=failed[i]['field'] or not isinstance(r.get('canonical'),str) or not r['canonical'].strip():
            errors.append('INVALID_REPAIR_SCHEMA');continue
        cs[i]={'field':r['field'],'canonical':r['canonical'].strip()}
    answer=execute(cs,exact,post) if not errors else []
    for i,c in enumerate(initial):
        if i not in failed:assert cs[i]==c
    return cs,answer,errors
