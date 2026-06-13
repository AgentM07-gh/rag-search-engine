import string


def token_split(input_str: string) -> list:

    punc_rem = str.maketrans("","",string.punctuation)

    token_str = input_str.lower().translate(punc_rem)
    token_list = token_str.split()
    token_list = [x for x in token_list if x != ""]

    return token_list