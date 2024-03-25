import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import threading
import gradio as gr
import re
from bs4 import BeautifulSoup
from markdown import markdown
import nltk
from nltk.tokenize import sent_tokenize
import string
import unicodedata
import time


nltk.download('punkt')
POST_ID = 0
REFERENDUM_TYPE = "referendums_v2"
VOTE_TYPE = "ReferendumV2"  # "Motion", "Fellowship", "Referendum", "ReferendumV2", "DemocracyProposal"
UPDATE_INTERVAL = 1800


def dot_product(u, v):
    res = np.dot(u, v)
    return res


def markdn_2_str(text):
    html = markdown(text)
    clean_text = ' '.join(BeautifulSoup(html, features="html.parser").findAll(string=True))
    markdown_link_pattern = re.compile(r'\[.*?\]\(.*?\)')
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean_text = re.sub(markdown_link_pattern, ' ', clean_text)  # remove markdown style links
    clean_text = re.sub(url_pattern, ' ', clean_text)  # remove regular links
    clean_text = clean_text.replace('\n', ' ')  # remove \n
    return clean_text


def get_sum(prop):
    key_word = "KSM"
    pattern = re.compile(r'(\d)')

    search_phrases = [
        "requests a total of 1500 KSM",
        "requests a total of 7450 $ (17 KSM)",
        "Requested amount: 3,333 KSM",
        "Requested funding 78,804 USD // 2770 KSM",
        "Requested KSM: 598"
    ]

    ref = model.encode("".join(search_phrases), convert_to_tensor=True)
    prop = unicodedata.normalize("NFKD", prop)
    prop = markdn_2_str(prop)
    sentences = sent_tokenize(prop)

    similarities = []
    for s in sentences:
        sentence_embedding = model.encode(s, convert_to_tensor=True)
        similarities.append(-dot_product(sentence_embedding, ref))
    max_similarity_index = np.argsort(similarities)

    sent = next((sentences[i] for i in max_similarity_index if "KSM" in sentences[i]), "None")

    s = re.split(r'(\s)', sent)
    s = [x.translate(str.maketrans('', '', string.punctuation)) if not pattern.search(x) else x for x in s]
    s = [x for x in s if x != ' ']
    s = [x for x in s if x != '']
    try:
        index_KSM = [idx for idx, val in enumerate(s) if "KSM" in val]
        for el in index_KSM:
            l = s[el - 1:el + 2]
            for x in l:
                if pattern.search(x):
                    return x
    except Exception:
        return None


def get_proposals():
    global POST_ID
    global df
    flag = True
    while flag:
        rn = requests.post(
            f"https://api.polkassembly.io/api/v1/posts/on-chain-post?proposalType={REFERENDUM_TYPE}&postId={POST_ID}",
            headers={"x-network": "kusama"})
        if rn.ok:
            print(POST_ID)
            proposal_data = rn.json()
            line = [proposal_data.get("content"), proposal_data.get("status"), get_sum(proposal_data.get("content"))]
            df.loc[POST_ID] = line
            POST_ID += 1
        else:
            event.set()
            flag = False


def get_embeddings():
    global df_emb
    for i in range(len(df)):
        df_emb.loc[i] = [model.encode(markdn_2_str(df.iloc[i]['content']))]


def update_proposals():
    global POST_ID
    global df
    flag = True
    while flag:
        rn = requests.post(
            f"https://api.polkassembly.io/api/v1/posts/on-chain-post?proposalType={REFERENDUM_TYPE}&postId={POST_ID}",
            headers={"x-network": "kusama"})
        if rn.ok:
            proposal_data = rn.json()
            line = [proposal_data.get("content"), proposal_data.get("status"), get_sum(proposal_data.get("content"))]
            df.loc[POST_ID] = line
            POST_ID += 1
        else:
            print('proposals updated at {t}'.format(t=time.strftime("%H:%M:%S", time.localtime())))
            event.set()
            flag = False


def update_embeddings():
    global df_emb
    while True:
        event.wait()
        print(POST_ID)
        print(len(df))

        if len(df) != len(df_emb):
            id_to_add = [x + len(df_emb) for x in range(len(df) - len(df_emb))]
            for i in id_to_add:
                df_emb.loc[i] = [model.encode(markdn_2_str(df.iloc[i]['content']))]
        else:
            event.clear()


def run_periodically():
    update_proposals()
    threading.Timer(UPDATE_INTERVAL, run_periodically).start()


def compare_proposals(prop, count):
    query_emb = model.encode(markdn_2_str(prop))
    new_df = pd.DataFrame(columns=['sim1'])
    new_df['sim1'] = df_emb.apply(lambda row: dot_product(row[0], query_emb), axis=1)
    best_match = np.argsort(-new_df['sim1'])[0:count]
    res = [df.iloc[x]['content'] for x in best_match]
    stat = [df.iloc[x]['status'] for x in best_match]
    ksm = [df.iloc[x]['ksm'] for x in best_match]
    title = [
        '''<span style="color:blue"><h2>Total KSM requested: {sum}, status: {status}, ID: {id}</h2></span> \n '''.format(
            sum=x, status=y, id=z) for x, y, z in zip(ksm, stat, best_match)]
    result = "\n ".join([a + b for a, b in zip(title, res)])
    return result


if __name__ == '__main__':
    print('start')
    event = threading.Event()
    model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
    print('model downloaded')

    df = pd.DataFrame(columns=['content', 'status', 'ksm'])
    df_emb = pd.DataFrame(columns=['content'])

    print('proposal collection start')
    get_proposals()
    print('proposals collected, embeddings calculation start')
    get_embeddings()

    POST_ID = len(df)

    update_thread = threading.Thread(target=run_periodically)  # background proposals update
    upd_emb_thread = threading.Thread(target=update_embeddings)  # background embeddings update

    update_thread.start()
    upd_emb_thread.start()

    print('gradio start')
    with gr.Blocks() as demo:
        gr.Markdown("<h1>Compare proposals</h1>")
        inpt = gr.Textbox(label="Input Proposal", lines=5, max_lines=12)
        dr = gr.Dropdown(label="Vote type",
                         choices=["Motion", "Fellowship", "Referendum", "ReferendumV2", "DemocracyProposal"],
                         value="ReferendumV2", interactive=True)
        slider = gr.Slider(label="Number of proposals to output", minimum=1, maximum=20, step=1, value=5,
                           interactive=True)
        btn = gr.Button("Find similar proposals")
        otpt = gr.Markdown("")
        btn.click(fn=compare_proposals, inputs=[inpt, slider], outputs=otpt)

    demo.launch(show_error=True)
