from typing import List

import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


def draw_text(
    text,
    weights=None,
    x_max=300,
    textsize=10,
    font='cour.ttf',
    cmap='Reds'
) -> Image:
    '''
    returns rendered text with weight shading
    '''

    font = ImageFont.truetype(font, textsize)
    # font = ImageFont.load_default()
    char_width = font.getsize(' ')[0]

    line_spacing = 1
    line_height = textsize*line_spacing
    line_length = x_max//char_width

    lines = []
    while len(text) > line_length:
        temp = text[:line_length]
        breakpoint = temp.rfind(' ')+1  # find breakpoint at word boundary
        if breakpoint == 0:
            raise Exception('Not enough space for text')
        lines.append(text[:breakpoint])
        text = text[breakpoint:]
    lines.append(text)

    box = (x_max, line_height*len(lines))
    img = Image.new('RGBA', box, color=(255, 255, 255, 255))
    d = ImageDraw.Draw(img)

    # bboxes
    bboxes = []
    top = 0
    for line in lines:
        left = -0.5*char_width
        for word in line.rstrip().split(' '):
            word_width = (len(word)+1)*char_width
            bboxes.append([word, top, left, top+line_height, left+word_width])
            left += word_width
        top += line_height

    if weights is not None:
        for w, (_, t, l, b, r) in zip(weights, bboxes):
            d.rectangle([l, t, r, b], fill=tuple(int(i*255)
                        for i in cm.get_cmap(cmap, 255)(w)))

    for idx, line in enumerate(lines):
        d.text((0, int(idx*line_height)), line, fill=(0, 0, 0, 255), font=font)

    return img


def visualize_text_saliency_for_subject(subject_path: str) -> None:
    subject_id = os.path.basename(subject_path)
    text_fixations_csv_path = f"{subject_path}/{subject_id}_words_views.csv"
    out_dir = f"{subject_path}/text_saliency"
    os.makedirs(out_dir, exist_ok=True)

    try:
        df_text_fixations = pd.read_csv(text_fixations_csv_path,
                                        dtype={'word': 'str'})
    except FileNotFoundError:
        return None

    for _, group in df_text_fixations.groupby(
            ['image_name', 'question_id']):
        questions_answer_text = []
        questions_answer_weights = []
        image_name = group.iloc[0]['image_name']
        question_id = group.iloc[0]['question_id']
        for questions_answer in ['Q', 'A', 'B', 'C']:
            group_in = group[group['question_answer'] == questions_answer]
            group_in = group_in.sort_values(by=['idx'])
            words = group_in['word'].tolist()
            weights = group_in['counts'].tolist()

            words.insert(0, f"{questions_answer})")
            try:
                text = " ".join(words)
            except TypeError:
                print(group_in)
                print(subject_id)
                print(words)
            questions_answer_text.append(text)
            weights.insert(0, 0)
            questions_answer_weights.extend(weights)

        questions_answer_weights = np.array(questions_answer_weights)
        questions_answer_weights = questions_answer_weights/questions_answer_weights.sum()
        img = draw_text(" ".join(questions_answer_text),
                        list(questions_answer_weights))
        img.save(
            f"{out_dir}/{image_name}_{question_id}.png", "PNG")
        img.close()


def visualize_text_saliency(subjects_path: List[str],) -> None:
    excluded = pd.read_csv("dataset/excluded.csv")
    excluded_subjects = excluded['subject_id'].unique()
    for subject_path in tqdm(subjects_path):
        if os.path.basename(subject_path) in excluded_subjects:
            continue
        visualize_text_saliency_for_subject(subject_path)
