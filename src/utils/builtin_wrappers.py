def contains_value(input_list, element):
    if element in input_list:
        return True
    else:
        return False

def number_of_elements_in_list(input_list):
    return len(input_list)

def remove_every_element_from_list(input_list):
    if input_list is None:
        return None
    else:
        input_list = input_list.clear()
        return None

def reverse_list(input_list):
    input_list.reverse()
    return input_list

def odds_from_list(input_list):
    return input_list[0::2]

def number_of_odds_in_list(input_list):
    c = len(input_list[0::2])
    return c

def contains_odd(input_list):
    for i in input_list:
        if i%2>0:
            return True
        else:
            return False

def second_largest_in_list(input_list):
    input_list.sort(reverse=True)
    return input_list[1]

def sum_of_elements_in_list(input_list):
    return sum(input_list)

def cumsum_list(input_list):
    length = len(input_list)
    cu_list = [sum(input_list[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

def element_wise_sum(input_list1, input_list2):
    output = []
    for i in range(0, len(input_list1)):
        output.append(input_list1[i]+input_list2[i])
    return output

def subset_of_list(input_list, start_index, end_index):
    return input_list[start_index:end_index+1]

def every_nth(input_list, step_size):
    return input_list[::step_size]

def only_unique_in_list(input_list):
    if len(set(input_list))==len(input_list):
        return True
    else:
        return False

def keep_unique(input_list):
    output = []
    for i in input_list:
        if i not in output:
            output.append(i)
        else:
            continue
    return output

def swap(input_list, first_index, second_index):
    input_list[first_index], input_list[second_index] = input_list[second_index], input_list[first_index]
    return input_list

def remove_element_by_value(input_list, value_to_remove):
    if value_to_remove in input_list:
        input_list.remove(value_to_remove)
        return input_list
    else:
        return input_list

def remove_element_by_index(input_list, index):
    if index <= len(input_list):
        input_list.remove(input_list[index])
        return input_list
    else:
        return input_list

def multiply_every_element(input_list, multiplier):
    output = []
    for i in input_list:
        x = i*multiplier
        output.append(x)
    return output

def remove_key(input_dict, key):
    if key in input_dict.keys():
        input_dict.pop(key)
        return input_dict
    else:
        return input_dict

def sort_by_key(input_dict):
    myKeys = list(input_dict.keys())
    myKeys.sort()
    sorted_dict = {i: input_dict[i] for i in myKeys}
    return  sorted_dict

def sum_in_dict(input_dict):
    return sum(input_dict.values())

def merge_two_dicts(input_dict1, input_dict2):
    input_dict1.update(input_dict2)
    return input_dict1

def merge_dicts():
    return False

def sort_list_by_parity(input_list):
    even = []
    odd = []
    my_dict = {}
    for i in input_list:
        if i%2==0:
            even.append(i)
        else:
            odd.append(i)
    my_dict['even'] = even
    my_dict['odd'] = odd
    return my_dict

def mean_by_key_value(input_dict):
    my_dict = {}
    for k in input_dict.keys():
        my_dict[k] = sum(input_dict[k])/len(input_dict[k])
    return my_dict

def count_frequency(input_list):
    frequency = {}
    for item in input_list:
        if item in frequency:
            frequency[item] += 1
        else:
            frequency[item] = 1
    return frequency