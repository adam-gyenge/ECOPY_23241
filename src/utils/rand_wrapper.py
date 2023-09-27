import random

random.seed(42)

def random_from_list(input_list):
    #return input_list[random.randint(0, len(input_list)-1)]
    return random.choice(input_list)


def random_sublist_from_list(input_list, number_of_elements):
    #return random.sample(input_list, number_of_elements)
    if number_of_elements > len(input_list):
        raise ValueError("number is longer then length of list")
    elif number_of_elements <= 0:
        raise ValueError("Num of elements has to be positive number")
    return random.choices(input_list, k=number_of_elements)


def random_from_string(input_string):
    return input_string[random.randint(0, len(input_string)-1)]


def hundred_small_random():
    output = []
    for i in range(100):
        output.append(random.random())
    return output


def hundred_large_random():
    out = []
    for i in range(100):
        out.append(random.randint(10, 1000))
    return out


def five_random_number_div_three():
    out = []
    while len(out)<5:
        c = random.randint(9, 1000)
        if c%3==0:
            out.append(c)
        else:
            continue
    return out


def random_reorder(input_list):
    return random.sample(input_list, len(input_list))


def uniform_one_to_five():
    return random.uniform(1, 6)
