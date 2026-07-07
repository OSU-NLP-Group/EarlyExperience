import os, re, json, requests, openai, argparse
from openai import OpenAI

SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer>Beijing</answer>."""

SEARCH_RE  = re.compile(r"<search>(.*?)</search>", re.S)
ANSWER_RE  = re.compile(r"<answer>(.*?)</answer>", re.S)

def search(query: str, retrieval_address) -> str:
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }
    results = requests.post(
        retrieval_address, json=payload, timeout=15
    ).json()['result']

    info = ''
    for idx, doc in enumerate(results[0]):
        content = doc['document']['contents']
        title, *rest = content.split('\n')
        body = '\n'.join(rest)
        info += f"Doc {idx+1}(Title: {title}) {body}\n"
    return info

def chat_once(client, user_prompt: str, model, temperature=0, max_tokens=4096) -> str:
    try:
        response = client.chat.completions.create(
                model       = model,
                messages    =   [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user",   "content": user_prompt},
                                ],
                temperature = temperature,
                max_tokens  = max_tokens,
            )
        return response.choices[0].message.content.strip()
    except:
        return "[FAILED: chat completion error]"

# ---------------- 主推理 ----------------
def answer_question(question,
                    client,
                    retrieval_address,
                    model,
                    max_turns = 10,
                    temperature = 0):
    user_prompt = f"Question: {question}"
    trace_text = ""
    dialog_log  = []

    for turn in range(1, max_turns + 1):
        assistant_reply = chat_once(client, user_prompt, model, temperature)
        dialog_log.append(("assistant", assistant_reply))
        user_prompt += assistant_reply
        trace_text  += assistant_reply

        if "[FAILED: chat completion error]" in assistant_reply:
            return "[FAILED: chat completion error]", dialog_log, trace_text
        ans_match = ANSWER_RE.search(assistant_reply)
        if ans_match:
            return ans_match.group(1).strip(), dialog_log, trace_text

        search_match = SEARCH_RE.search(assistant_reply)
        if not search_match:
            return "[FAILED: no search & no answer]", dialog_log, trace_text

        query = search_match.group(1).strip()

        query_text = search(query,retrieval_address).strip("\n")

        info_block = f"<information>{query_text}</information>"

        dialog_log.append(("user", info_block))
        user_prompt += info_block
        trace_text  += info_block

    return "[FAILED: max_turns exceeded]", dialog_log, trace_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="Path to store output results")
    parser.add_argument("--retrieval_address", required=True, help="Search API endpoint")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible model API base URL")
    parser.add_argument("--temperature", type=float, default = 0.0, help="Temperature for the model")
    parser.add_argument("--input_path", default="./data/evaluation_data/6_OOD_dataset.json")
    parser.add_argument("--eval_data", default="musique")
    args = parser.parse_args()

    # client = OpenAI(api_key="0",base_url="http://c0801.ten.osc.edu:8890/v1")
    # retrieval_address = "http://c0818.ten.osc.edu:10000/retrieve"

    client = OpenAI(api_key="0",base_url=args.base_url)
    retrieval_address = args.retrieval_address
    model = ""

    input_path = "./data/evaluation_data/6_OOD_dataset.json"
    multihop_input_path = "./data/evaluation_data/multihop_ood.json"
    musique_input_path = "./data/evaluation_data/sampled_musique_validation.json"
    output_path = args.output_path


    with open(musique_input_path, "r", encoding="utf-8") as f_in:
        musique_samples = [json.loads(line) for line in f_in]

    if args.eval_data == "musique":
        samples = musique_samples
    elif args.eval_data == "multihop":
        multihop_ood_samples = json.load(open(multihop_input_path, encoding="utf-8"))
        samples = musique_samples + multihop_ood_samples
    else:
        ood_samples = json.load(open(input_path, encoding="utf-8"))
        samples = musique_samples + ood_samples

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if 'id' in obj:
                    data_source = obj["data_source"]
                    id_text = {obj['id']}
                    done_ids.add(f"{id_text}_{data_source}")
    for sample in samples:
        data_source = sample["data_source"]
        id_text = {sample['id']}
        if f"{id_text}_{data_source}" in done_ids:
            continue
        q = sample['question']
        ans, log, trace = answer_question(q,client,retrieval_address,model, temperature=args.temperature)
        sample["predicted_answer"] = ans
        sample["dialog_log"] = log
        sample["trace_text"] = trace
        with open(output_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(sample) + '\n')
    


