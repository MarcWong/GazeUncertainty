from typing import Dict, Tuple

import argparse
import demjson
import regex
import re

import pandas as pd

from dataset.dataset_io import save_submit

TEvent = Tuple[str, str, str, str, str]
TTimestamp_to_submit = Dict[str, Tuple[TEvent, Dict]]

JSON_REGEX_STRING = regex.compile(r'\{(?:[^{}]|(?R))*\}')

EVENT_STAGE = None


def parse_event_string(event_string: str, time: str) -> Tuple:
    """Parses an event string like:

    QA button clicked, question No1: Fri Nov 26 2021 22:42:39 GMT+0100
     (Central European Standard Time)

    for important data.

    Careful! Manipulatey global variable EVENT_STAGE.

    :param event_string: The event string to parse.
    :type event_string: str
    :param time: Time that the event took place in unix format.
    :type time: str
    :raises ValueError: When an event string is not matchable.
    :return: Tuple.
    :rtype: Tuple
    """
    global EVENT_STAGE
    if event_string.startswith("next button clicked"):
        target = re.findall(r'task:[0-9]+,', event_string)
        assert(len(target) == 1)
        target = target[0][:-1]
        return ("next button clicked", target,
                target.split(":")[1],
                target.split(":")[0],
                EVENT_STAGE,
                time)
    elif event_string.startswith("QA button clicked"):
        target = re.findall(r'question No[0-9]+:', event_string)
        assert(len(target) == 1)
        target = target[0][:-1]
        return ("QA button clicked", target,
                target.split(" No")[1],
                target.split(" No")[0],
                EVENT_STAGE,
                time)
    elif event_string.startswith("Entering recognition stage"):
        EVENT_STAGE = "recognition"
        return ("Entering recognition stage", None, None, None, EVENT_STAGE,
                time)
    elif event_string.startswith("Entering recall stage"):
        EVENT_STAGE = "recall"
        return ("Entering recall stage", None, None, None, EVENT_STAGE, time)
    elif event_string.startswith("Submitting study"):
        return ("Submitting study", None, None, None, EVENT_STAGE, time)
    else:
        raise ValueError(f'Event string {event_string} could not be matched.')


def parse_event_file(
    event_file: str
) -> Tuple[pd.DataFrame, TTimestamp_to_submit]:
    """Parse an event file and return two dictionaries:

    1. One that maps timestamps to all event strings
    2. One that maps timestamps to all submit strings and their submit value

    :param event_file: The debougout event file from the web survey.
    :type event_file: str
    :return: (Event dictionary, Submit dictionary)
    :rtype: Tuple[pd.DataFrame, TTimestamp_to_submit]
    """
    global EVENT_STAGE
    parsed_events = []
    parsed_submit = []

    with open(event_file, 'r') as file:
        EVENT_STAGE = None
        content = file.read()
        submitted = JSON_REGEX_STRING.findall(content)
        submitted = [re.sub(r'\s+', '', sub) for sub in submitted]
        submitted = [re.sub(r'origin:,', 'origin:null,', sub)
                     for sub in submitted]
        submitted = [demjson.decode(sub) for sub in submitted]
        content = JSON_REGEX_STRING.sub('', content)
        lines = content.splitlines()
        lines = list(filter(lambda x: not re.match(
            r'^\s*$|----.*----', x), lines))

        for i in range(0, len(lines), 2):
            if lines[i].startswith("Submitting"):
                event = lines[i]
                event_time_stamp = lines[i+1]
                parsed_submit.append((
                    parse_event_string(event, event_time_stamp),
                    submitted.pop(0)
                ))
            else:
                event = lines[i]
                event_time_stamp = lines[i+1]
                parsed_events.append(parse_event_string(
                    event, event_time_stamp))
    events_df = pd.DataFrame.from_records(
        parsed_events,
        columns=['event_name', 'target', 'target_id',
                 'target_name', 'event_stage', 'timestamp']
    )
    return events_df, parsed_submit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_file", type=str, default=None)
    args = vars(parser.parse_args())

    events, submits = parse_event_file(args['event_file'])
    print(events.head())
    for submit in submits.values():
        save_submit(submit[1])
