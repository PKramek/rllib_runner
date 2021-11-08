from typing import Dict, Iterable, Tuple


def split_dictionary(dictionary: Dict, keys: Iterable) -> Tuple[Dict, Dict]:
    """
    Splits dictionary into two. Key, value pairs, where key is in keys iterable will be in one resulting dictionary and
    all the rest of the pairs will be in second dictionary

    :param dictionary: Dictionary to be split
    :type dictionary: Dict
    :param keys: Keys used to split dictionary. All the key, value pairs with key in that iterable will be in only one
    resulting dictionary
    :type keys: Iterable
    :return: Two dictionaries, first contains all the key, value pairs where key was not preset in given keys iterable
    :rtype: Tuple[Dict, Dict]
    """
    assert set(keys) <= set(dictionary.keys()), 'All keys must be in original dictionary'

    without_keys = {key: value for (key, value) in dictionary.items() if key not in keys}
    with_keys = {key: value for (key, value) in dictionary.items() if key in keys}

    return without_keys, with_keys


def get_sub_dictionary(dictionary: Dict, keys: Iterable) -> Dict:
    """
    Return dictionary containing only those key, value pairs which have key in keys iterable

    :param dictionary: Dictionary to get sub dictionary from
    :type dictionary: Dict
    :param keys: Subset of original dictionary keys
    :type keys: Iterable
    :return: Sub dictionary
    :rtype: Dictionary
    """
    assert set(keys) <= set(dictionary.keys()), "All keys must be in original dictionary"

    with_keys = {key: value for (key, value) in dictionary.items() if key in keys}

    return with_keys
