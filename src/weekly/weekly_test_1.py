# r8qokx
# Gyenge Ádám

def evens_from_list(input_list):
    return input_list[1::2]

def every_element_is_odd(input_list):
    for i in input_list:
        if i % 2 == 0:
            return False
        else:
            return True

def kth_largest_in_list(input_list, kth_largest):
    if kth_largest<=len(input_list):
        input_list.sort()
        return input_list[kth_largest]
    else:
        return 'kth_largest is higher than length of list'


def cumavg_list(input_list):
    out = []
    output = []
    for i in input_list:
        output.append(i)
        out.append(sum(output)/len(output))
    return out


def element_wise_multiplication(input_list1, input_list2 ):
    output = []
    for i in range(0, len(input_list1)):
        output.append(input_list1[i] * input_list2[i])
    return output


def merge_lists(*lists):
    output = []
    for i in lists:
        output = output + i
    return output


def squared_odds(input_list):
    output = []
    for i in input_list[0::2]:
        if i % 2 > 0:
            output.append(i**2)
        else:
            continue
    return output


def reverse_sort_by_key(input_dict):
    myKeys = list(input_dict.keys())
    myKeys.sort(reverse=True)
    sorted_dict = {i: input_dict[i] for i in myKeys}
    return sorted_dict


def sort_list_by_divisibility(input_list):
    by_2 = []
    by_5 = []
    by_25 = []
    by_none = []
    my_dict = {}
    for i in input_list:
        if i % 2 == 0 and i % 5 == 0:
            by_25.append(i)
        elif i % 2 == 0:
            by_2.append(i)
        elif i % 5 == 0:
            by_5.append(i)
        else:
            by_none.append(i)
    my_dict['by_two'] = by_2
    my_dict['by_five'] = by_5
    my_dict['by_two_and_five'] = by_25
    my_dict['by_none'] = by_none
    return my_dict