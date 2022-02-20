from typing import Dict
import pandas as pd


def match_subject_answers_to_question_views(df_question_views: pd.DataFrame,
                                            submit: Dict) -> pd.DataFrame:
    """Attches answers to the dataframe.

    :param df_question_views: Views of questions.
    :type df_question_views: pd.DataFrame
    :param submit: The submit json.
    :type submit: Dict
    :return: The dataframe with answers attached.
    :rtype: pd.DataFrame
    """
    qa_answers = submit['results']['outputs']['qa_answers']
    assert(len(df_question_views) == len(qa_answers))
    df_question_views['subject_answer'] = qa_answers
    return df_question_views
