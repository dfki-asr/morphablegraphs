""" Functions for setting the values in dictionary based on JSONPath.
    Existing libraries do not support changing values (03/30/17)

    Some functions are copied from the following sources:
    https://github.com/kennknowles/python-jsonpath-rw/issues/2
    http://stackoverflow.com/questions/2103071/determine-the-type-of-a-value-which-is-represented-as-string-in-python
    http://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
"""


import distutils
import re
from anim_utils.utilities.log import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, set_log_mode

TYPE_CONVERTER = {
    "int": int,
    "float": float,
    "bool": lambda value: bool(distutils.util.strtobool(value)),
    "str": str
}


def update_json(data, path, value):
    """ Update JSON dictionnary PATH with VALUE. Return updated JSON"""
    try:
        first = next(path)
        data[first] = update_json(data[first], path, value)
        return data
    except StopIteration:
        return value


def get_type_of_string(str_value):
    """ Identiy the type of a string value """
    for type, test in list(TYPE_CONVERTER.items()):
        try:
            v = test(str_value)
            if str_value == str(v):
                return type
        except ValueError:
            continue
    # No match
    return "str"


def get_path_from_string(json_path_str):
    """ Convert a JSONPath string into a list of keys for access to the dictionary"""
    temp_path_list = json_path_str.split(".")[1:]
    path_list = []
    for idx, key in enumerate(temp_path_list):
        matches = list(re.finditer("\[-?\d+\]", key))
        if len(matches) > 0:
            for match_idx, match in enumerate(matches):
                span = match.span()
                index = int(key[span[0] + 1:span[1] - 1])

                if match_idx == 0:
                    short_key = key[:span[0]]
                    path_list.append(short_key)
                path_list.append(index)
        else:
            path_list.append(key)
    return path_list


def search_for_path(data, json_path_str):
    """ Get the reference to the value in a dictionary given a JSONPath """
    path = get_path_from_string(json_path_str)
    current = data
    for key in path:
        try:
            current = current[key]
        except:
            return None
    return current


def update_data_using_jsonpath(data, expressions, split_str="="):
    """ Takes a dictionary and a list of expressions in the form JSONPath=value, e.g. "$.write_log=True".
        Expressions and values should not contain the split_str="="
    """
    for expr in expressions:
        expr_t = expr.split(split_str)
        json_path_str = expr_t[0]
        value = expr_t[1]

        match = search_for_path(data, json_path_str)
        if match is not None:
            before = match
            path_list = iter(get_path_from_string(json_path_str))

            value_type = get_type_of_string(value)
            update_json(data, path_list, TYPE_CONVERTER[value_type](value))

            match = search_for_path(data, json_path_str)
            message = "set value of " + json_path_str + " from " + str(before) + " to " + str(match) + " with " + str(type(match))
            write_message_to_log(message, LOG_MODE_DEBUG)

        else:
            write_message_to_log("Warning: Did not find JSONPath " + json_path_str + " in data", LOG_MODE_ERROR)


if __name__ == "__main__":
    test_data = {
        "model_data": "motion_primitives_quaternion_PCA95_unity-integration-1.0.0",
        "port": 8888,
        "write_log": True,
        "log_level": 1,
        "list_test": [{"ab":1},{"ab":3}]
    }

    import argparse
    set_log_mode(LOG_MODE_DEBUG)
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("-set", nargs='+', default=[], help="JSONPath expression")
    args = parser.parse_args()

    update_data_using_jsonpath(test_data, args.set)
