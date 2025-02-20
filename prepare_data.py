import os, json, re
from tqdm import tqdm

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
# for split in ['train', 'test']:
#     examples = json.load(open('/root/autodl-tmp/math-project/TAL-SCQ5K/need_cot_generate_data.json', 'r'))
examples = read_jsonl('/root/autodl-tmp/dataset/chatharuhi-118k/Haruhi54K.jsonl')


# results = []
f = open('/root/autodl-tmp/dataset/chatharuhi-118k/Haruhi54K_train.jsonl', 'w')
f_ = open('/root/autodl-tmp/dataset/chatharuhi-118k/debug.txt', 'w')
for ex in tqdm(examples, total=len(examples)):
    # 确定role_name
    if ex['context'].startswith('I want you to act like'):
        pattern = re.compile(r'I want you to act like.* from')
        select_string = pattern.findall(ex['context'])[0]
        role_name = ' '.join(select_string.split()[6:-1])
    elif ex['context'].startswith('Please be aware that your codename in this  conversation is'):
        pattern = re.compile(r'Please be aware that your codename in this  conversation is\s{0,2}[\'|‘][\u4e00-\u9fa5]{2,4}[\'|’]')
        select_string = pattern.findall(ex['context'])[0]
        role_name = ''.join(re.compile(r'[\u4e00-\u9fa5]').findall(select_string.split()[-1].strip()))
    elif ex['context'].startswith('你正在扮演'):
        if '于谦' in ex['context']:
            role_name = '于谦'
        else:
            role_name = '李云龙'
    # else:
    #     print(ex['context'])
    # print(role_name)
    try:
        assert 'Classic scenes for the role are as follows:' in ex['context']
        system_prompt = ex['context'].split('Classic scenes for the role are as follows:')[0]
    except:
        print('system_prompt: ', system_prompt)

    conversations = '\n'.join([ex['context'], ex['target']]).split('Classic scenes for the role are as follows:')[1].split('\n###')
    try:
        assert len(conversations) > 1
    except:
        print('conversations: ', conversations)

    
    # pattern_user = re.compile(r'\n?.+[:|：]\s?「.+」')
    
    for con in conversations: # \n阿朱:「什么剑谱？在那里？先给我瞧瞧是真还是假的。」
        if con.strip():
            message = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt.strip()
                    }]}
            # res = [item.strip() for item in pattern_user.findall(con)]
            res = [item.strip() for item in con.split('\n') if item.strip()]
            start_index = 0
            end_index = 0
            while end_index < len(res):
                if res[end_index].startswith(role_name+':') or res[end_index].startswith(role_name+'：'):
                    try:
                        assert end_index > 0
                    except:
                        # print('con: ', con)
                        f_.write(con + '\n')
                        message['messages'].append({
                            "role": "user",
                            "content": ''
                        })

                        message['messages'].append({
                            "role": "assistant",
                            "content": res[end_index].replace(role_name+':', '').replace(role_name, '').replace(role_name+'：', '')
                        })
                        
                        start_index = end_index + 1
                        end_index += 1
                        continue

                    if end_index == start_index:
                        # role_name连续回复,则拼接起来
                        message['messages'][-1]["content"] += res[end_index].replace(role_name+':', '').replace(role_name, '').replace(role_name+'：', '')
                    else:
                        message['messages'].append({
                            "role": "user",
                            "content": ' '.join(res[start_index:end_index]).strip()
                        })

                        message['messages'].append({
                            "role": "assistant",
                            "content": res[end_index].replace(role_name+':', '').replace(role_name, '').replace(role_name+'：', '')
                        })
                    start_index = end_index + 1
                end_index += 1
            f.write(json.dumps(message, ensure_ascii=False) + '\n')
            # print(json.dumps(message, ensure_ascii=False, indent=2))
    # res.append(message)
f.close()
