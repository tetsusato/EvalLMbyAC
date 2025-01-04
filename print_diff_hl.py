import difflib

def print_diff_hl(ground_truth, target):
    """
    文字列の差異をハイライト表示する
    """
    color_dic = {
        'red':'\033[31m',
        'yellow':'\033[43m',  
        'end':'\033[0m'
    }
    
    d = difflib.Differ()
    diffs = d.compare(ground_truth, target)
    
    result = ''
    for diff in diffs:
        status, _, character = list(diff)
        if status == '-':
            character = color_dic['red'] + character + color_dic['end']
        elif status == '+':
            character = color_dic['yellow'] + character + color_dic['end']
        else:
            pass
        result += character
    
    print(f"ground truth : {ground_truth}")
    print(f"target string: {target}")
    print(f"diff result  : {result}")

def get_diff_hl(ground_truth, target):
    """
    文字列の差異をハイライト表示する
    """
    color_dic = {
        'red':'\033[31m',
        'yellow':'\033[43m',  
        'end':'\033[0m'
    }
    
    d = difflib.Differ()
    diffs = d.compare(ground_truth, target)
    
    result = ''
    for diff in diffs:
        status, _, character = list(diff)
        if status == '-':
            character = color_dic['red'] + character + color_dic['end']
        elif status == '+':
            character = color_dic['yellow'] + character + color_dic['end']
        else:
            pass
        result += character
    
    return result


if __name__=="__main__":
    ground_truth = "abcde"
    target = "a ydef"
    print_diff_hl(ground_truth, target)
