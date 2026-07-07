import json
import collections
import re
from tqdm import tqdm
import openai
import string
import os
from collections import defaultdict

# ------------------ Text Normalization and Scoring ------------------

def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# ------------------ GPT-4o Judge Prompt ------------------

JUDGE_PROMPT_TEMPLATE = """
Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. 
The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. 
Respond with True if the prediction is correct and False otherwise.

Question: {question}
Golden Answer: {gold}
Predicted Answer: {pred}
"""


def judge_answer_gpt(question, gold_answer, pred_answer, model="gpt-4o-mini"):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold=gold_answer,
        pred=pred_answer,
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = response.choices[0].message.content.strip().lower()
        return reply
    except Exception as e:
        print(f"Error during GPT-judging: {e}")
        return None


def gpt_vote(args):
    """
    单个样本的 GPT 判定.
    args = (question, golds, pred, model)
    返回 (gpt_correct: bool, gpt_raw_reply: str|None)
    """
    question, golds, pred, model = args
    client = openai.OpenAI()                    

    for gold in golds:                          
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, gold=gold, pred=pred
        )
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            reply = rsp.choices[0].message.content.strip().lower()
        except Exception as e:
            return False, None

        if "true" in reply:
            return True, reply
    return False, reply          

# ------------------ Evaluation Loop (parallel GPT) ------------------
import multiprocessing as mp


def evaluation(file_path, use_gpt=True, max_samples=2000, evaluation_name="",out_path="",
               gpt_workers=40, gpt_model = "gpt-4o-mini", cal_data = ["musique"]):

    total_em = total_f1 = total_gpt = 0
    total_dialog_turns = 0
    records   = []
    gpt_tasks = []

    source_stats = defaultdict(lambda: {
        "n": 0, "em": 0, "f1": 0, "gpt": 0,
        "dialog_turns": 0                      # ➋ 新增：分来源累计 assistant 轮数
    })

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            if max_samples and len(records) >= max_samples:
                break
            data = json.loads(line)

            question = data["question"]
            pred     = data.get("predicted_answer", data.get("predicated_answer"))
            if data.get("golden_answers"):
                golds = data["golden_answers"] if isinstance(data["golden_answers"], list) \
                                            else [data["golden_answers"]]
            else:
                golds = []

                if "answer" in data:
                    golds.append(data["answer"])

                golds.extend(data.get("answer_aliases", []))


            em = max(compute_exact(g, pred) for g in golds)
            f1 = max(compute_f1(g, pred)   for g in golds)


            if data["data_source"] not in cal_data:
                continue
            src = data.get("data_source", "unknown")
            stat = source_stats[src]
            stat["n"]  += 1
            stat["em"] += em
            stat["f1"] += f1

            assistant_turns = sum(
                1 for t in data.get("dialog_log", [])
                if isinstance(t, list) and t and t[0] == "assistant"
            )
            total_dialog_turns += assistant_turns
            source_stats[src]["dialog_turns"] += assistant_turns

            if use_gpt:
                gpt_tasks.append((question, golds, pred, gpt_model))

            records.append({
                "raw": data, "question": question, "golds": golds, "pred": pred,
                "em": em, "f1": f1, "gpt_correct": None, "gpt_reply": None,
                "assistant_turns": assistant_turns,
                "src": src,
            })

            total_em += em
            total_f1 += f1
            

    if use_gpt:
        if gpt_workers in (0, 1):
            gpt_results = map(gpt_vote, gpt_tasks)
        else:
            with mp.Pool(processes=gpt_workers) as pool:
                gpt_results = pool.map(gpt_vote, gpt_tasks)

        for rec, (ok, reply) in zip(records, gpt_results):
            rec["gpt_correct"] = ok
            rec["gpt_reply"]   = reply

            if ok:
                total_gpt += 1
            src = rec["src"]
            source_stats[src]["gpt"] += int(ok)

    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in records:
            out = rec["raw"]
            out.update({
                "em":  rec["em"],
                "f1":  rec["f1"],
                "ground_truth": rec["golds"],
                **(
                    {"gpt_judged": rec["gpt_correct"],
                     "gpt_response": rec["gpt_reply"]}
                    if use_gpt else {}
                )
            })
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(records)
    print("\n=== Overall Results ===")
    print(f"evaluation_name: {evaluation_name}")
    print(f"Total Samples: {n}")
    print(f"Average EM:   {total_em / n:.4f}")
    print(f"Average F1:   {total_f1 / n:.4f}")
    print(f"Average assistant turns: {total_dialog_turns / n:.2f}")
    if use_gpt:
        print(f"GPT Accuracy: {total_gpt / n:.4f}")

    print("\n=== Results by data_source ===")
    header = f"{'source':<25} | {'#':>6} | {'EM':>6} | {'F1':>6} | {'GPT':>6} | {'TURN':>6}"
    print(header)
    print("-" * len(header))
    for src, s in sorted(source_stats.items(), key=lambda x: -x[1]["n"]):
        avg_em  = s["em"]  / s["n"]
        avg_f1  = s["f1"]  / s["n"]
        avg_gpt = s["gpt"] / s["n"] if use_gpt else 0
        avg_dt  = s["dialog_turns"] / s["n"]
        print(f"{src:<25} | {s['n']:>6} | {avg_em:6.3f} | {avg_f1:6.3f} | {avg_gpt:6.3f} | {avg_dt:6.2f}")

# ------------------ Run ------------------

if __name__ == "__main__":

    rollout = 1

    num = 3500

    cal_data = [
                "musique",
                "2wikimultihopqa",
                "hotpotqa",
                "bamboogle",
                ]

    model_results = "Llama-3.1-8B-Instruct"
    # model_results = "Llama-3.2-3B-Instruct"
    # model_results = "Qwen2.5-7B-Instruct"

    evaluation_name_list = [
        # "base_model",
        "posttrain_decision_data_1.0e-5",
        "posttrain_decision_wm_summary_data_1.0e-5",
        "posttrain_decision_wm_summary_1:1_data_1.0e-5",
        "posttrain_decision_wm_query_data_1.0e-5",
        "posttrain_reasoning_v6_decision_data_1.0e-5",
        "posttrain_reasoning_v5_decision_data_1.0e-5",
        "posttrain_reasoning_v4_decision_data_1.0e-5",
        
    ]
    for evaluation_name in evaluation_name_list:

        try:
            jinput_path = f"./results/{model_results}/{evaluation_name}.json"  # Replace with your file path
            output_path = f"./results/{model_results}/output_with_metrics/{evaluation_name}.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            evaluation(jinput_path, use_gpt=False,max_samples=num, evaluation_name=evaluation_name, out_path=output_path, cal_data=cal_data)
        except Exception as e:
            pass
            # print(f"Error: {e}")
            # print(f"Error: {evaluation_name}")