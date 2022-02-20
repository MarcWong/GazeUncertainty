import pandas as pd

from bs4 import BeautifulSoup

QUESTION_IDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']


def add_entries(res, elements, image_name, question_answer, question_id):
    idx = 0
    for elem in elements:
        if elem.text.strip() == '':
            continue
        res['id'].append(elem.get('id'))
        res['word'].append(elem.text.strip())
        res['image_name'].append(image_name)
        res['question_id'].append(question_id)
        res['question_answer'].append(question_answer)
        res['idx'].append(idx)
        idx += 1


def parse_question_html(experiment_group):
    qas = [event for event in experiment_group if event if "QA" in event]
    res = {'word': [], 'id': [], 'image_name': [],
           'question_answer': [], 'question_id': [], 'idx': []}
    for qa in qas:
        image_name = qa['QA']['name']
        for id in QUESTION_IDS:
            add_entries(res, BeautifulSoup(
                qa['QA'][id]['question'], 'html.parser').find_all('span'),
                image_name, 'Q', id)
            add_entries(res, BeautifulSoup(
                qa['QA'][id]['A'], 'html.parser').find_all('span'),
                image_name, 'A', id)
            add_entries(res, BeautifulSoup(
                qa['QA'][id]['B'], 'html.parser').find_all('span'),
                image_name, 'B', id)
            add_entries(res, BeautifulSoup(
                qa['QA'][id]['C'], 'html.parser').find_all('span'),
                image_name, 'C', id)
            add_entries(res, BeautifulSoup(
                qa['QA'][id]['D'], 'html.parser').find_all('span'),
                image_name, 'D', id)
    return pd.DataFrame.from_records(res)


def filter_fixatios_to_words(
    df_fixations,
    df_baseline_text_boxes,
    experiment_group
) -> pd.DataFrame:
    df_words = parse_question_html(experiment_group)
    df_words_with_boxes = pd.merge(df_words, df_baseline_text_boxes, on='id')
    df_fixations = df_fixations[['image_name', 'axp', 'ayp', 'dur']]
    df = pd.merge(df_words_with_boxes, df_fixations, on='image_name')
    df = df[(df['axp'] < df['right'])
            & (df['left'] < df['axp'])
            & (df['top'] < df['ayp'])
            & (df['ayp'] < df['bottom'])]
    all_words = df_words_with_boxes[[
        'id', 'image_name', 'question_answer', 'question_id', 'word', 'idx']]
    df = df[[
        'id', 'image_name', 'question_answer', 'question_id', 'word', 'idx']]
    df = pd.concat([all_words, df])
    df = df.value_counts(
        subset=['id', 'image_name', 'question_answer',
                'question_id', 'word', 'idx']
    ).to_frame('counts').reset_index()
    df = df.sort_values(by=['image_name', 'question_id', 'idx'])
    df['counts'] = df['counts'] - 1
    return df
